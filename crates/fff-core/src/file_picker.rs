//! Core file picker: filesystem indexing, background watching, and fuzzy search.
//!
//! [`FilePicker`] is the central component of fff-search. It:
//!
//! 1. **Indexes** a directory tree in a background thread, collecting every
//!    non-ignored file into a path-sorted `Vec<FileItem>`.
//! 2. **Watches** the filesystem via the `notify` crate, applying
//!    create/modify/delete events to the index in real time.
//! 3. **Owns files**: Provides a values for search and provides a good entry point for
//!    fuzzy search and live grep
//!
//! # Lifecycle
//!
//! ```text
//!   new_with_shared_state()
//!     │
//!     ├─> background scan thread ──> populates SharedPicker
//!     └─> file-system watcher    ──> live updates SharedPicker
//!
//!   fuzzy_search()   <── static, borrows &[FileItem]
//!   grep()           <── static, borrows &[FileItem] (live content search)
//!   trigger_rescan() <── synchronous re-index
//!   cancel()         <── shuts down background work
//! ```
//!
//! # Thread Safety
//!
//! `FilePicker` itself is **not** `Sync`!
//! all concurrent access goes through [`SharedPicker`](crate::SharedPicker) .
//! The background scanner and watcher acquire write locks only when mutating
//! the file index, so read-heavy search workloads rarely contend.

use crate::background_watcher::BackgroundWatcher;
use crate::bigram_filter::{BigramFilter, BigramIndexBuilder, BigramOverlay};
use crate::error::Error;
use crate::frecency::FrecencyTracker;
use crate::git::GitStatusCache;
use crate::grep::{GrepResult, GrepSearchOptions, grep_search};
use crate::ignore::non_git_repo_overrides;
use crate::query_tracker::QueryTracker;
use crate::score::match_and_score_files;
use crate::shared::{SharedFrecency, SharedPicker};
use crate::types::{
    ContentCacheBudget, DirItem, FileItem, PaginationArgs, ScoringContext, SearchResult,
};
use fff_query_parser::FFFQuery;
use git2::{Repository, Status, StatusOptions};
use rayon::prelude::*;
use std::fmt::Debug;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, LazyLock,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};
use std::time::SystemTime;
use tracing::{Level, debug, error, info, warn};

/// Dedicated thread pool for background work (scan, warmup, bigram build).
/// Uses fewer threads than the global rayon pool so Neovim's event loop
/// and search queries can still get CPU time.
static BACKGROUND_THREAD_POOL: LazyLock<rayon::ThreadPool> = LazyLock::new(|| {
    let total = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    let bg_threads = total.saturating_sub(2).max(1);
    info!(
        "Background pool: {} threads (system has {})",
        bg_threads, total
    );
    rayon::ThreadPoolBuilder::new()
        .num_threads(bg_threads)
        .thread_name(|i| format!("fff-bg-{i}"))
        .build()
        .expect("failed to create background rayon pool")
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FFFMode {
    #[default]
    Neovim,
    Ai,
}

impl FFFMode {
    pub fn is_ai(self) -> bool {
        self == FFFMode::Ai
    }
}

/// Configuration for a single fuzzy search invocation.
///
/// Passed to [`FilePicker::fuzzy_search`] to control threading, pagination,
/// and scoring behavior.
#[derive(Debug, Clone, Copy, Default)]
pub struct FuzzySearchOptions<'a> {
    pub max_threads: usize,
    pub current_file: Option<&'a str>,
    pub project_path: Option<&'a Path>,
    pub combo_boost_score_multiplier: i32,
    pub min_combo_count: u32,
    pub pagination: PaginationArgs,
}

#[derive(Debug, Clone)]
struct FileSync {
    git_workdir: Option<PathBuf>,
    /// All files: `files[..base_count]` are sorted by path (base index, used
    /// for binary search and bigram);
    ///
    /// `files[base_count..]` are overflow files added since the last full reindex.
    /// Deletions in the base use tombstones (`is_deleted = true`) to keep bigram indices stable.
    files: Vec<FileItem>,
    /// Number of base files (the sorted prefix used for binary search / bigram).
    base_count: usize,
    /// Sorted directory table. Each entry is a unique parent directory of at
    /// least one file in `files`. Sorted by absolute path for O(log n) lookup.
    /// Built during `walk_filesystem` and used for directory picker mode,
    /// per-directory stats, and as a fast replacement for `extract_watch_dirs`.
    dirs: Vec<DirItem>,
    /// SIMD-aligned chunk arena holding deduplicated directory paths.
    /// Each unique directory's path is stored as a sequence of 16-byte chunks.
    /// FileItem dir_ptrs point into this arena — keeping it alive is critical.
    #[allow(dead_code)]
    dir_chunk_arena: Vec<crate::simd_path::SimdChunk>,
    /// Packed contiguous filename arena. FileItem filename_ptrs point here.
    #[allow(dead_code)]
    filename_arena: Vec<u8>,
    /// Overflow arena for watcher-added file paths. Each entry is a separate
    /// heap allocation holding `dir_rel_path ++ filename` (with SIMD padding)
    /// for one overflow file. Using per-file allocations ensures that growing
    /// the Vec doesn't invalidate pointers of previously-added files.
    overflow_arena: Vec<Box<[u8]>>,
    /// Compressed bigram inverted index built during the post-scan phase.
    /// Lives here so that replacing `FileSync` on rescan automatically drops
    /// the stale index (bigram file indices are positions in `files`).
    bigram_index: Option<Arc<BigramFilter>>,
    /// Overlay tracking file mutations since the bigram index was built.
    bigram_overlay: Option<Arc<parking_lot::RwLock<BigramOverlay>>>,
    /// Bigram index built from file relative paths (for fuzzy search pre-filtering).
    /// Much smaller than the content bigram index but dramatically reduces the
    /// number of files that reach the expensive SIMD Smith-Waterman matching.
    path_bigram_index: Option<Arc<BigramFilter>>,
}

impl FileSync {
    fn new() -> Self {
        Self {
            files: Vec::new(),
            base_count: 0,
            dirs: Vec::new(),
            dir_chunk_arena: Vec::new(),
            filename_arena: Vec::new(),
            overflow_arena: Vec::new(),
            git_workdir: None,
            bigram_index: None,
            bigram_overlay: None,
            path_bigram_index: None,
        }
    }

    /// Get all files (base + overflow). The base portion `[..base_count]` is
    /// sorted by path; the overflow tail is unsorted.
    #[inline]
    fn files(&self) -> &[FileItem] {
        &self.files
    }

    /// Get the overflow portion (files added since last full reindex).
    #[inline]
    fn overflow_files(&self) -> &[FileItem] {
        &self.files[self.base_count..]
    }

    #[allow(dead_code)]
    fn get_file(&self, index: usize) -> Option<&FileItem> {
        self.files.get(index)
    }

    /// Get mutable file at index (works for both base and overflow)
    #[inline]
    fn get_file_mut(&mut self, index: usize) -> Option<&mut FileItem> {
        self.files.get_mut(index)
    }

    /// Find file index by path using binary search on the sorted base portion.
    #[inline]
    fn find_file_index(&self, path: &Path) -> Result<usize, usize> {
        let path_str = path.as_os_str().to_string_lossy();

        // Look up the directory index for the parent of this path
        let parent_end = path_str.rfind('/').unwrap_or(path_str.len());
        let parent_path = &path_str[..parent_end];
        let filename = &path_str[parent_end.saturating_add(1)..];

        // Binary search dirs to find the parent directory index
        let dir_idx = match self
            .dirs
            .binary_search_by(|d| d.path_str().cmp(parent_path))
        {
            Ok(idx) => idx as u32,
            Err(_) => return Err(0), // directory not found
        };

        // Binary search files by (parent_dir, filename) — same order as the sort
        self.files[..self.base_count].binary_search_by(|f| {
            f.parent_dir_index()
                .cmp(&dir_idx)
                .then_with(|| f.file_name().cmp(filename))
        })
    }

    /// Find a file in the overflow portion by relative path (linear scan).
    /// Returns the absolute index into `files`.
    ///
    /// the overflowed items are not ordered so we can not use binary search
    fn find_overflow_index(&self, rel_path: &str) -> Option<usize> {
        self.files[self.base_count..]
            .iter()
            .position(|f| f.relative_path() == rel_path)
            .map(|pos| self.base_count + pos)
    }

    /// Get file count
    #[inline]
    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.files.len()
    }

    /// Insert a file at position. Simple - no HashMap to maintain!
    fn insert_file(&mut self, position: usize, file: FileItem) {
        self.files.insert(position, file);
    }

    /// Remove file at index. Simple - no HashMap to maintain!
    #[allow(dead_code)]
    fn remove_file(&mut self, index: usize) {
        if index < self.files.len() {
            self.files.remove(index);
        }
    }

    /// Remove files matching predicate from both base and overflow.
    /// Returns number of files removed. Adjusts `base_count` accordingly.
    fn retain_files<F>(&mut self, mut predicate: F) -> usize
    where
        F: FnMut(&FileItem) -> bool,
    {
        let initial_len = self.files.len();
        // Count how many base files survive.
        let base_retained = self.files[..self.base_count]
            .iter()
            .filter(|f| predicate(f))
            .count();
        self.files.retain(predicate);
        self.base_count = base_retained;
        initial_len - self.files.len()
    }

    /// Insert a file in sorted order (by path).
    /// Returns true if inserted, false if file already exists.
    fn insert_file_sorted(&mut self, file: FileItem, base_path: &Path) -> bool {
        let abs_path = file.absolute_path(base_path);
        match self.find_file_index(&abs_path) {
            Ok(_) => false, // File already exists
            Err(position) => {
                self.insert_file(position, file);
                true
            }
        }
    }
}

impl FileItem {
    pub fn new(path: PathBuf, base_path: &Path, git_status: Option<Status>) -> (Self, String) {
        let metadata = std::fs::metadata(&path).ok();
        Self::new_with_metadata(path, base_path, git_status, metadata.as_ref())
    }

    /// Create a FileItem using pre-fetched metadata to avoid a redundant stat syscall.
    /// Returns `(FileItem, String)` — the String keeps the path data alive until
    /// `build_path_arena` copies the relative path into the arena and calls `repoint_path`.
    pub fn new_with_metadata(
        path: PathBuf,
        base_path: &Path,
        git_status: Option<Status>,
        metadata: Option<&std::fs::Metadata>,
    ) -> (Self, String) {
        let relative_path = pathdiff::diff_paths(&path, base_path)
            .unwrap_or_else(|| path.clone())
            .to_string_lossy()
            .into_owned();

        let (size, modified) = match metadata {
            Some(metadata) => {
                let size = metadata.len();
                let modified = metadata
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_secs());

                (size, modified)
            }
            None => (0, 0),
        };

        let is_binary = is_known_binary_extension(&path);

        let path_string = path.to_string_lossy().into_owned();
        let relative_start = (path_string.len() - relative_path.len()) as u16;
        let filename_start = path_string
            .rfind(std::path::MAIN_SEPARATOR)
            .map(|i| i + 1)
            .unwrap_or(relative_start as usize) as u16;

        let item = Self::new_raw(
            &path_string,
            relative_start,
            filename_start,
            size,
            modified,
            git_status,
            is_binary,
        );
        (item, path_string)
    }

    /// Create a FileItem and write its dir + filename directly into arenas.
    ///
    /// Dirs are deduplicated via `dir_dedup` — files in the same directory
    /// share the same bytes in `dir_arena`. Filenames are always appended.
    ///
    /// **Important**: dir_ptr and filename_ptr are set to OFFSET values (not real
    /// pointers) because the arenas may reallocate during the parallel walk.
    /// Call `fixup_arena_ptrs` after the walk to convert offsets to real pointers.
    pub fn new_into_arenas(
        path: PathBuf,
        base_path: &Path,
        git_status: Option<Status>,
        metadata: Option<&std::fs::Metadata>,
        dir_arena: &mut Vec<u8>,
        dir_dedup: &mut ahash::AHashMap<Box<[u8]>, (u32, u16)>,
        filename_arena: &mut Vec<u8>,
    ) -> Self {
        let (size, modified) = match metadata {
            Some(metadata) => {
                let size = metadata.len();
                let modified = metadata
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_secs());
                (size, modified)
            }
            None => (0, 0),
        };

        let is_binary = is_known_binary_extension(&path);

        let rel = pathdiff::diff_paths(&path, base_path).unwrap_or_else(|| path.clone());
        let rel_str = rel.to_string_lossy();
        let fname_offset_in_rel = rel_str.rfind('/').map(|i| i + 1).unwrap_or(0);

        let dir_part = &rel_str[..fname_offset_in_rel];
        let fname_part = &rel_str[fname_offset_in_rel..];

        // Dedup dir: hash the dir bytes, reuse existing offset if found.
        let dir_hash = {
            use std::hash::{Hash, Hasher};
            let mut hasher = ahash::AHasher::default();
            dir_part.hash(&mut hasher);
            hasher.finish()
        };

        let (dir_offset, dir_len) = if let Some(&(off, len)) = dir_dedup.get(&dir_hash) {
            // Verify it's not a hash collision (extremely rare with 64-bit hash)
            debug_assert_eq!(
                &dir_arena[off as usize..off as usize + len as usize],
                dir_part.as_bytes(),
            );
            (off as usize, len)
        } else {
            let off = dir_arena.len();
            dir_arena.extend_from_slice(dir_part.as_bytes());
            let len = dir_part.len() as u16;
            dir_dedup.insert(dir_hash, (off as u32, len));
            (off, len)
        };

        // Filenames are always unique — append directly.
        let fname_offset = filename_arena.len();
        filename_arena.extend_from_slice(fname_part.as_bytes());

        Self::from_arena_ptrs(
            dir_offset as *const u8,
            dir_len,
            fname_offset as *const u8,
            fname_part.len() as u16,
            size,
            modified,
            git_status,
            is_binary,
        )
    }

    pub fn update_frecency_scores(
        &mut self,
        tracker: &FrecencyTracker,
        base_path: &Path,
        mode: FFFMode,
    ) -> Result<(), Error> {
        self.access_frecency_score =
            tracker.get_access_score(&self.absolute_path(base_path), mode) as i16;
        self.modification_frecency_score =
            tracker.get_modification_score(self.modified, self.git_status, mode) as i16;

        Ok(())
    }
}

/// Options for creating a [`FilePicker`].
pub struct FilePickerOptions {
    pub base_path: String,
    pub warmup_mmap_cache: bool,
    pub mode: FFFMode,
    /// Explicit cache budget. When `None`, the budget is auto-computed from
    /// the repo size after the initial scan completes.
    pub cache_budget: Option<ContentCacheBudget>,
    /// When `false`, `new_with_shared_state` skips the background file watcher.
    /// Files are still scanned, warmed up, and bigram-indexed.
    pub watch: bool,
}

impl Default for FilePickerOptions {
    fn default() -> Self {
        Self {
            base_path: ".".into(),
            warmup_mmap_cache: false,
            mode: FFFMode::default(),
            cache_budget: None,
            watch: true,
        }
    }
}

pub struct FilePicker {
    pub mode: FFFMode,
    pub base_path: PathBuf,
    pub is_scanning: Arc<AtomicBool>,
    sync_data: FileSync,
    cache_budget: Arc<ContentCacheBudget>,
    has_explicit_cache_budget: bool,
    watcher_ready: Arc<AtomicBool>,
    scanned_files_count: Arc<AtomicUsize>,
    background_watcher: Option<BackgroundWatcher>,
    warmup_mmap_cache: bool,
    watch: bool,
    cancelled: Arc<AtomicBool>,
    // This is a soft lock that we use to prevent rescan be triggered while the
    // bigram indexing is in progress. This allows to keep some of the unsafe magic
    // relying on the immutabillity of the files vec after the index without worrying
    // that the vec is going to be dropped before the indexing is finished
    //
    // In addition to that rescan is likely triggered by something unnecessary
    // before the indexing is finished it means that fff is dogfooded the index either
    // by the UI rendering preview or simply by walking the directory. Which is not good anyway
    post_scan_busy: Arc<AtomicBool>,
}

impl std::fmt::Debug for FilePicker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilePicker")
            .field("base_path", &self.base_path)
            .field("sync_data", &self.sync_data)
            .field("is_scanning", &self.is_scanning.load(Ordering::Relaxed))
            .field(
                "scanned_files_count",
                &self.scanned_files_count.load(Ordering::Relaxed),
            )
            .finish_non_exhaustive()
    }
}

impl FilePicker {
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Convert an absolute path to a relative path string (relative to base_path).
    /// Returns None if the path doesn't start with base_path.
    fn to_relative_path<'a>(&self, path: &'a Path) -> Option<&'a str> {
        path.strip_prefix(&self.base_path)
            .ok()
            .and_then(|p| p.to_str())
    }

    pub fn need_warmup_mmap_cache(&self) -> bool {
        self.warmup_mmap_cache
    }

    pub fn mode(&self) -> FFFMode {
        self.mode
    }

    pub fn cache_budget(&self) -> &ContentCacheBudget {
        &self.cache_budget
    }

    pub fn bigram_index(&self) -> Option<&BigramFilter> {
        self.sync_data.bigram_index.as_deref()
    }

    pub fn bigram_overlay(&self) -> Option<&parking_lot::RwLock<BigramOverlay>> {
        self.sync_data.bigram_overlay.as_deref()
    }

    /// Get the path bigram index for fuzzy search pre-filtering.
    pub fn path_bigram_index(&self) -> Option<&BigramFilter> {
        self.sync_data.path_bigram_index.as_deref()
    }

    pub fn get_file_mut(&mut self, index: usize) -> Option<&mut FileItem> {
        self.sync_data.get_file_mut(index)
    }

    pub fn set_bigram_index(&mut self, index: BigramFilter, overlay: BigramOverlay) {
        self.sync_data.bigram_index = Some(Arc::new(index));
        self.sync_data.bigram_overlay = Some(Arc::new(parking_lot::RwLock::new(overlay)));
    }

    pub fn git_root(&self) -> Option<&Path> {
        self.sync_data.git_workdir.as_deref()
    }

    /// Get all indexed files sorted by path.
    /// Note: Files are stored sorted by PATH for efficient insert/remove.
    /// For frecency-sorted results, use search() which sorts matched results.
    pub fn get_files(&self) -> &[FileItem] {
        self.sync_data.files()
    }

    pub fn get_overflow_files(&self) -> &[FileItem] {
        self.sync_data.overflow_files()
    }

    /// Get the directory table (sorted by path).
    pub fn get_dirs(&self) -> &[DirItem] {
        &self.sync_data.dirs
    }

    /// Actual heap bytes used: (dir_chunk_arena, filename_arena, overflow_arena).
    pub fn arena_bytes(&self) -> (usize, usize, usize) {
        let chunk = self.sync_data.dir_chunk_arena.len() * 16;
        let fname = self.sync_data.filename_arena.len();
        let overflow = self
            .sync_data
            .overflow_arena
            .iter()
            .map(|b| b.len())
            .sum::<usize>();
        (chunk, fname, overflow)
    }

    /// Extracts all unique ancestor directories from the indexed file list.
    /// Uses the pre-built directory table when available (O(d) where d = unique dirs),
    /// falling back to the old traversal for overflow files.
    #[tracing::instrument(level = "debug", skip(self))]
    pub fn extract_watch_dirs(&self) -> Vec<PathBuf> {
        let dir_table = &self.sync_data.dirs;

        if !dir_table.is_empty() {
            // Fast path: just collect PathBufs from the dir table.
            // The dir table already contains all unique parent directories.
            // We also need ancestor directories (parents of parents) for the
            // watcher to work. Walk up from each dir to the base.
            let base = self.base_path.as_path();
            let mut all_dirs = Vec::with_capacity(dir_table.len() * 2);
            let mut seen = std::collections::HashSet::with_capacity(dir_table.len() * 2);

            for dir_item in dir_table {
                let mut current = dir_item.as_path().to_path_buf();
                while current.as_path() != base {
                    if !seen.insert(current.clone()) {
                        break; // already visited this and all its ancestors
                    }
                    all_dirs.push(current.clone());
                    if !current.pop() {
                        break;
                    }
                }
            }

            return all_dirs;
        }

        // Fallback: old traversal for cases where dir table is empty
        let files = self.sync_data.files();
        let base = self.base_path.as_path();
        let mut dirs = Vec::with_capacity(files.len() / 4);
        let mut current = self.base_path.clone();

        for file in files {
            let abs = file.absolute_path(base);
            let Some(parent) = abs.parent() else {
                continue;
            };
            if parent == current.as_path() {
                continue;
            }

            while current.as_path() != base && !parent.starts_with(&current) {
                current.pop();
            }

            let Ok(remainder) = parent.strip_prefix(&current) else {
                continue;
            };
            for component in remainder.components() {
                current.push(component);
                dirs.push(current.clone());
            }
        }

        dirs
    }

    /// Create a new FilePicker from options.
    /// Always prefer new_with_shared_state for the consumer application, use this only if you know
    /// what you are doing. This won't spawn the backgraound watcher and won't walk the file tree.
    pub fn new(options: FilePickerOptions) -> Result<Self, Error> {
        let path = PathBuf::from(&options.base_path);
        if !path.exists() {
            error!("Base path does not exist: {}", options.base_path);
            return Err(Error::InvalidPath(path));
        }
        if path.parent().is_none() {
            error!("Refusing to index filesystem root: {}", path.display());
            return Err(Error::FilesystemRoot(path));
        }

        let has_explicit_budget = options.cache_budget.is_some();
        let initial_budget = options.cache_budget.unwrap_or_default();

        Ok(FilePicker {
            background_watcher: None,
            base_path: path,
            cache_budget: Arc::new(initial_budget),
            cancelled: Arc::new(AtomicBool::new(false)),
            has_explicit_cache_budget: has_explicit_budget,
            is_scanning: Arc::new(AtomicBool::new(false)),
            mode: options.mode,
            post_scan_busy: Arc::new(AtomicBool::new(false)),
            scanned_files_count: Arc::new(AtomicUsize::new(0)),
            sync_data: FileSync::new(),
            warmup_mmap_cache: options.warmup_mmap_cache,
            watch: options.watch,
            watcher_ready: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Create a picker, place it into the shared handle, and spawn background
    /// indexing + file-system watcher. This is the default entry point.
    pub fn new_with_shared_state(
        shared_picker: SharedPicker,
        shared_frecency: SharedFrecency,
        options: FilePickerOptions,
    ) -> Result<(), Error> {
        let picker = Self::new(options)?;

        info!(
            "Spawning background threads: base_path={}, warmup={}, mode={:?}",
            picker.base_path.display(),
            picker.warmup_mmap_cache,
            picker.mode,
        );

        let warmup = picker.warmup_mmap_cache;
        let watch = picker.watch;
        let mode = picker.mode;

        picker.is_scanning.store(true, Ordering::Release);

        let scan_signal = Arc::clone(&picker.is_scanning);
        let watcher_ready = Arc::clone(&picker.watcher_ready);
        let synced_files_count = Arc::clone(&picker.scanned_files_count);
        let cancelled = Arc::clone(&picker.cancelled);
        let post_scan_busy = Arc::clone(&picker.post_scan_busy);
        let path = picker.base_path.clone();

        {
            let mut guard = shared_picker.write()?;
            *guard = Some(picker);
        }

        spawn_scan_and_watcher(
            path,
            scan_signal,
            watcher_ready,
            synced_files_count,
            warmup,
            watch,
            mode,
            shared_picker,
            shared_frecency,
            cancelled,
            post_scan_busy,
        );

        Ok(())
    }

    /// Synchronous filesystem scan — populates `self` with indexed files.
    ///
    /// Use this when you need direct access to the picker without shared state:
    /// ```ignore
    /// let mut picker = FilePicker::new(options)?;
    /// picker.collect_files()?;
    /// // picker.get_files() is now populated
    /// ```
    pub fn collect_files(&mut self) -> Result<(), Error> {
        self.is_scanning.store(true, Ordering::Relaxed);
        self.scanned_files_count.store(0, Ordering::Relaxed);

        let empty_frecency = SharedFrecency::default();
        let walk = walk_filesystem(
            &self.base_path,
            &self.scanned_files_count,
            &empty_frecency,
            self.mode,
        )?;

        self.sync_data = walk.sync;

        // Recalculate cache budget based on actual file count (unless
        // the caller provided an explicit budget via FilePickerOptions).
        if !self.has_explicit_cache_budget {
            let file_count = self.sync_data.files().len();
            self.cache_budget = Arc::new(ContentCacheBudget::new_for_repo(file_count));
        } else {
            self.cache_budget.reset();
        }

        // Apply git status synchronously.
        if let Ok(Some(git_cache)) = walk.git_handle.join() {
            for file in self.sync_data.files.iter_mut() {
                file.git_status = git_cache.lookup_status(&file.absolute_path(&self.base_path));
            }
        }

        self.is_scanning.store(false, Ordering::Relaxed);
        Ok(())
    }

    /// Start the background file-system watcher.
    ///
    /// The picker must already be placed into `shared_picker` (the watcher
    /// needs the shared handle to apply live updates). Call after
    /// [`collect_files`](Self::collect_files) or after an initial scan.
    pub fn spawn_background_watcher(
        &mut self,
        shared_picker: &SharedPicker,
        shared_frecency: &SharedFrecency,
    ) -> Result<(), Error> {
        let git_workdir = self.sync_data.git_workdir.clone();
        let watch_dirs = self.extract_watch_dirs();
        let watcher = BackgroundWatcher::new(
            self.base_path.clone(),
            git_workdir,
            shared_picker.clone(),
            shared_frecency.clone(),
            self.mode,
            watch_dirs,
        )?;
        self.background_watcher = Some(watcher);
        self.watcher_ready.store(true, Ordering::Release);
        Ok(())
    }

    /// Perform fuzzy search on files with a pre-parsed query.
    ///
    /// The query should be parsed using [`FFFQuery`]::parse() before calling
    /// this function. If a [`QueryTracker`] is provided, the search will
    /// automatically look up the last selected file for this query and apply
    /// combo-boost scoring.
    ///
    pub fn fuzzy_search<'a, 'q>(
        files: &'a [FileItem],
        query: &'q FFFQuery<'q>,
        query_tracker: Option<&QueryTracker>,
        options: FuzzySearchOptions<'q>,
        path_bigram_index: Option<&BigramFilter>,
    ) -> SearchResult<'a> {
        let max_threads = if options.max_threads == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        } else {
            options.max_threads
        };

        debug!(
            raw_query = ?query.raw_query,
            pagination = ?options.pagination,
            ?max_threads,
            current_file = ?options.current_file,
            "Fuzzy search",
        );

        let total_files = files.len();
        let location = query.location;

        // Get effective query for max_typos calculation (without location suffix)
        let effective_query = match &query.fuzzy_query {
            fff_query_parser::FuzzyQuery::Text(t) => *t,
            fff_query_parser::FuzzyQuery::Parts(parts) if !parts.is_empty() => parts[0],
            _ => query.raw_query.trim(),
        };

        // small queries with a large number of results can match absolutely everything
        let max_typos = (effective_query.len() as u16 / 4).clamp(2, 6);
        // Look up the last file selected for this query (combo-boost scoring)
        let last_same_query_entry =
            query_tracker
                .zip(options.project_path)
                .and_then(|(tracker, project_path)| {
                    tracker
                        .get_last_query_entry(
                            query.raw_query,
                            project_path,
                            options.min_combo_count,
                        )
                        .ok()
                        .flatten()
                });

        let context = ScoringContext {
            query,
            max_typos,
            max_threads,
            project_path: options.project_path,
            current_file: options.current_file,
            last_same_query_match: last_same_query_entry,
            combo_boost_score_multiplier: options.combo_boost_score_multiplier,
            min_combo_count: options.min_combo_count,
            pagination: options.pagination,
            path_bigram_index,
        };

        let time = std::time::Instant::now();
        let (items, scores, total_matched) = match_and_score_files(files, &context);

        info!(
            ?query,
            completed_in = ?time.elapsed(),
            total_matched,
            returned_count = items.len(),
            pagination = ?options.pagination,
            "Fuzzy search completed",
        );

        SearchResult {
            items,
            scores,
            total_matched,
            total_files,
            location,
        }
    }

    /// Perform a live grep search across indexed files with a pre-parsed query.
    pub fn grep(&self, query: &FFFQuery<'_>, options: &GrepSearchOptions) -> GrepResult<'_> {
        let overlay_guard = self.sync_data.bigram_overlay.as_ref().map(|o| o.read());
        grep_search(
            self.get_files(),
            query,
            options,
            self.cache_budget(),
            self.sync_data.bigram_index.as_deref(),
            overlay_guard.as_deref(),
            Some(&self.cancelled),
            &self.base_path,
        )
    }

    /// Like [`grep`](Self::grep) but ignores the bigram overlay.
    /// Useful for testing that the overlay is actually contributing results.
    pub fn grep_without_overlay(
        &self,
        query: &FFFQuery<'_>,
        options: &GrepSearchOptions,
    ) -> GrepResult<'_> {
        grep_search(
            self.get_files(),
            query,
            options,
            self.cache_budget(),
            self.sync_data.bigram_index.as_deref(),
            None,
            Some(&self.cancelled),
            &self.base_path,
        )
    }

    // Returns an ongoing or finisshed scan progress
    pub fn get_scan_progress(&self) -> ScanProgress {
        let scanned_count = self.scanned_files_count.load(Ordering::Relaxed);
        let is_scanning = self.is_scanning.load(Ordering::Relaxed);
        ScanProgress {
            scanned_files_count: scanned_count,
            is_scanning,
            is_watcher_ready: self.watcher_ready.load(Ordering::Relaxed),
            is_warmup_complete: self.sync_data.bigram_index.is_some(),
        }
    }

    /// Update git statuses for files, using the provided shared frecency tracker.
    pub fn update_git_statuses(
        &mut self,
        status_cache: GitStatusCache,
        shared_frecency: &SharedFrecency,
    ) -> Result<(), Error> {
        debug!(
            statuses_count = status_cache.statuses_len(),
            "Updating git status",
        );

        let mode = self.mode;
        let bp = self.base_path.clone();
        let frecency = shared_frecency.read()?;
        status_cache
            .into_iter()
            .try_for_each(|(path, status)| -> Result<(), Error> {
                if let Some(file) = self.get_mut_file_by_path(&path) {
                    file.git_status = Some(status);
                    if let Some(ref f) = *frecency {
                        file.update_frecency_scores(f, &bp, mode)?;
                    }
                } else {
                    error!(?path, "Couldn't update the git status for path");
                }
                Ok(())
            })?;

        Ok(())
    }

    pub fn update_single_file_frecency(
        &mut self,
        file_path: impl AsRef<Path>,
        frecency_tracker: &FrecencyTracker,
    ) -> Result<(), Error> {
        let path = file_path.as_ref();
        let rel = self.to_relative_path(path).unwrap_or("");
        let index = self
            .sync_data
            .find_file_index(path)
            .ok()
            .or_else(|| self.sync_data.find_overflow_index(rel));
        if let Some(index) = index
            && let Some(file) = self.sync_data.get_file_mut(index)
        {
            file.update_frecency_scores(frecency_tracker, &self.base_path, self.mode)?;
        }

        Ok(())
    }

    pub fn get_file_by_path(&self, path: impl AsRef<Path>) -> Option<&FileItem> {
        self.sync_data
            .find_file_index(path.as_ref())
            .ok()
            .and_then(|index| self.sync_data.files().get(index))
    }

    pub fn get_mut_file_by_path(&mut self, path: impl AsRef<Path>) -> Option<&mut FileItem> {
        let path = path.as_ref();
        let rel = self.to_relative_path(path).unwrap_or("");
        // Check sorted base first (O(log n)), then overflow tail (O(k)).
        let index = self
            .sync_data
            .find_file_index(path)
            .ok()
            .or_else(|| self.sync_data.find_overflow_index(rel));
        index.and_then(|i| self.sync_data.get_file_mut(i))
    }

    /// Add a file to the picker's files in sorted order (used by background watcher)
    pub fn add_file_sorted(&mut self, file: FileItem) -> Option<&FileItem> {
        let path = file.absolute_path(&self.base_path);

        if self.sync_data.insert_file_sorted(file, &self.base_path) {
            // File was inserted, look it up
            self.sync_data
                .find_file_index(&path)
                .ok()
                .and_then(|idx| self.sync_data.get_file_mut(idx))
                .map(|file_mut| &*file_mut) // Convert &mut to &
        } else {
            // File already exists
            warn!(
                "Trying to insert a file that already exists: {}",
                path.display()
            );
            self.sync_data
                .find_file_index(&path)
                .ok()
                .and_then(|idx| self.sync_data.get_file_mut(idx))
                .map(|file_mut| &*file_mut) // Convert &mut to &
        }
    }

    #[tracing::instrument(skip(self), name = "timing_update", level = Level::DEBUG)]
    pub fn on_create_or_modify(&mut self, path: impl AsRef<Path> + Debug) -> Option<&FileItem> {
        let path = path.as_ref();

        // Clone the overlay Arc upfront so we can access it independently of
        // the mutable borrow on sync_data.files (just a refcount bump).
        let overlay = self.sync_data.bigram_overlay.clone();

        // Check if this is a tombstoned base file being re-created.
        if let Ok(pos) = self.sync_data.find_file_index(path) {
            let file = self.sync_data.get_file_mut(pos)?;

            if file.is_deleted() {
                // Resurrect tombstoned file.
                file.set_deleted(false);
                debug!(
                    "on_create_or_modify: resurrected tombstoned file at index {}",
                    pos
                );
            }

            debug!(
                "on_create_or_modify: file EXISTS at index {}, updating metadata",
                pos
            );

            let modified = match std::fs::metadata(path) {
                Ok(metadata) => metadata
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok()),
                Err(e) => {
                    error!("Failed to get metadata for {}: {}", path.display(), e);
                    None
                }
            };

            if let Some(modified) = modified {
                let modified = modified.as_secs();
                if file.modified < modified {
                    file.modified = modified;
                    file.invalidate_mmap(&self.cache_budget);
                }
            }

            // Update the bigram overlay for this modified file.
            if let Some(ref overlay) = overlay
                && let Ok(content) = std::fs::read(path)
            {
                overlay.write().modify_file(pos, &content);
            }

            return Some(&*file);
        }

        // Check overflow for existing added files.
        let rel_path = self.to_relative_path(path).unwrap_or("");
        if let Some(abs_pos) = self.sync_data.find_overflow_index(rel_path) {
            let file = &mut self.sync_data.files[abs_pos];
            let modified = std::fs::metadata(path)
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok());
            if let Some(modified) = modified {
                let modified = modified.as_secs();
                if file.modified < modified {
                    file.modified = modified;
                    file.invalidate_mmap(&self.cache_budget);
                }
            }
            return Some(&self.sync_data.files[abs_pos]);
        }

        // New file — append to overflow tail (preserves base indices for bigram).
        debug!(
            "on_create_or_modify: file NEW, appending to overflow (base: {}, overflow: {})",
            self.sync_data.base_count,
            self.sync_data.overflow_files().len(),
        );

        let (file_item, _temp_path) = FileItem::new(path.to_path_buf(), &self.base_path, None);

        // Build a single per-file buffer holding dir + filename as 16B chunks.
        // Each overflow file gets its own Box<[u8]>, so growing the Vec never
        // invalidates pointers of previously-added files.
        //
        // Layout: [ dir chunks (16B aligned) ][ fname chunks (overlap + fname + pad) ]
        let dir = file_item.dir_str();
        let fname = file_item.file_name();
        let dir_padded = (dir.len() + 15) & !15;
        let overlap = dir.len() % 16; // dir tail bytes that bleed into fname's first 16B window
        let fname_bridged = overlap + fname.len();
        let fname_padded = (fname_bridged + 15) & !15;
        let total = dir_padded + fname_padded;

        let mut buf = vec![0u8; total];
        buf[..dir.len()].copy_from_slice(dir.as_bytes());
        // Fill fname chunk area: overlap prefix + filename
        if overlap > 0 {
            let dir_overlap_start = dir.len() - overlap;
            buf[dir_padded..dir_padded + overlap]
                .copy_from_slice(&dir.as_bytes()[dir_overlap_start..]);
        }
        buf[dir_padded + overlap..dir_padded + overlap + fname.len()]
            .copy_from_slice(fname.as_bytes());

        let boxed: Box<[u8]> = buf.into_boxed_slice();
        let dir_ptr = boxed.as_ptr();
        let fname_ptr = unsafe { boxed.as_ptr().add(dir_padded) };
        self.sync_data.overflow_arena.push(boxed);

        self.sync_data.files.push(file_item);
        // SAFETY: Box<[u8]> is heap-allocated and won't move when the Vec grows.
        // Pointers into it remain valid for the lifetime of this FileSync.
        unsafe {
            let file = self.sync_data.files.last_mut().unwrap();
            file.repoint_dir(dir_ptr);
            file.repoint_filename(fname_ptr);
        }

        self.sync_data.files.last()
    }

    /// Tombstone a file instead of removing it, keeping base indices stable.
    pub fn remove_file_by_path(&mut self, path: impl AsRef<Path>) -> bool {
        let path = path.as_ref();
        match self.sync_data.find_file_index(path) {
            Ok(index) => {
                let file = &mut self.sync_data.files[index];
                file.set_deleted(true);
                file.invalidate_mmap(&self.cache_budget);
                if let Some(ref overlay) = self.sync_data.bigram_overlay {
                    overlay.write().delete_file(index);
                }
                true
            }
            Err(_) => {
                // Check overflow for added files — these can be removed directly
                // since they aren't in the base bigram index.
                let rel = self.to_relative_path(path).unwrap_or("");
                if let Some(abs_pos) = self.sync_data.find_overflow_index(rel) {
                    self.sync_data.files.remove(abs_pos);
                    true
                } else {
                    false
                }
            }
        }
    }

    // TODO make this O(n)
    pub fn remove_all_files_in_dir(&mut self, dir: impl AsRef<Path>) -> usize {
        let dir_path = dir.as_ref();
        let dir_rel = self.to_relative_path(dir_path).unwrap_or("").to_string();
        let dir_prefix = if dir_rel.is_empty() {
            String::new()
        } else {
            format!("{}/", dir_rel)
        };
        // Use the safe retain_files method which maintains both indices
        self.sync_data.retain_files(|file| {
            !file.relative_path().starts_with(&dir_prefix) && file.relative_path() != dir_rel
        })
    }

    /// Use this to prevent any substantial background threads from acquiring the locks
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn stop_background_monitor(&mut self) {
        if let Some(mut watcher) = self.background_watcher.take() {
            watcher.stop();
        }
    }

    /// Spawn a background thread to rebuild the bigram index after rescan.
    pub(crate) fn spawn_post_rescan_rebuild(&self, shared_picker: SharedPicker) -> bool {
        if !self.warmup_mmap_cache || self.cancelled.load(Ordering::Relaxed) {
            return false;
        }

        let post_scan_busy = Arc::clone(&self.post_scan_busy);
        let cancelled = Arc::clone(&self.cancelled);
        let auto_budget = !self.has_explicit_cache_budget;

        post_scan_busy.store(true, Ordering::Release);

        std::thread::spawn(move || {
            let phase_start = std::time::Instant::now();

            // Scale cache budget if not explicitly configured.
            if auto_budget
                && !cancelled.load(Ordering::Acquire)
                && let Ok(mut guard) = shared_picker.write()
                && let Some(ref mut picker) = *guard
                && !picker.has_explicit_cache_budget
            {
                let file_count = picker.sync_data.files().len();
                picker.cache_budget = Arc::new(ContentCacheBudget::new_for_repo(file_count));
            }

            // Take a snapshot of files + budget while holding a brief read lock.
            // SAFETY: post_scan_busy blocks trigger_rescan from replacing
            // sync_data, so the Vec backing this slice stays alive.
            let files_snapshot = if !cancelled.load(Ordering::Acquire) {
                shared_picker.read().ok().and_then(|guard| {
                    guard.as_ref().map(|picker| {
                        let files = picker.sync_data.files();
                        let ptr = files.as_ptr();
                        let len = files.len();
                        let budget = Arc::clone(&picker.cache_budget);
                        let static_files: &[FileItem] =
                            unsafe { std::slice::from_raw_parts(ptr, len) };
                        (static_files, budget)
                    })
                })
            } else {
                None
            };

            if let Some((files, budget)) = files_snapshot {
                // Warmup mmap caches.
                if !cancelled.load(Ordering::Acquire) {
                    let t = std::time::Instant::now();
                    warmup_mmaps(files, &budget);
                    info!(
                        "Rescan warmup completed in {:.2}s (cached {} files, {} bytes)",
                        t.elapsed().as_secs_f64(),
                        budget.cached_count.load(Ordering::Relaxed),
                        budget.cached_bytes.load(Ordering::Relaxed),
                    );
                }

                // Build bigram index (lock-free).
                if !cancelled.load(Ordering::Acquire) {
                    let t = std::time::Instant::now();
                    info!(
                        "Rescan: starting bigram index build for {} files...",
                        files.len()
                    );
                    let (index, content_binary) = build_bigram_index(files, &budget);
                    info!(
                        "Rescan: bigram index ready in {:.2}s",
                        t.elapsed().as_secs_f64()
                    );

                    // Brief write lock to store the index.
                    if let Ok(mut guard) = shared_picker.write()
                        && let Some(ref mut picker) = *guard
                    {
                        for &idx in &content_binary {
                            if let Some(file) = picker.sync_data.get_file_mut(idx) {
                                file.set_binary(true);
                            }
                        }

                        let base_count = picker.sync_data.base_count;
                        picker.sync_data.bigram_index = Some(Arc::new(index));
                        picker.sync_data.bigram_overlay = Some(Arc::new(parking_lot::RwLock::new(
                            BigramOverlay::new(base_count),
                        )));
                    }
                }
            }

            post_scan_busy.store(false, Ordering::Release);
            info!(
                "Rescan post-scan warmup + bigram total: {:.2}s",
                phase_start.elapsed().as_secs_f64(),
            );
        });

        true
    }

    pub fn trigger_rescan(&mut self, shared_frecency: &SharedFrecency) -> Result<(), Error> {
        if self.is_scanning.load(Ordering::Relaxed) {
            debug!("Scan already in progress, skipping trigger_rescan");
            return Ok(());
        }

        // The post-scan warmup + bigram phase holds a raw pointer into the
        // current files Vec. Replacing sync_data now would free that memory.
        // Skip — the background watcher will retry on the next event.
        if self.post_scan_busy.load(Ordering::Acquire) {
            debug!("Post-scan bigram build in progress, skipping rescan");
            return Ok(());
        }

        self.is_scanning.store(true, Ordering::Relaxed);
        self.scanned_files_count.store(0, Ordering::Relaxed);

        let walk_result = walk_filesystem(
            &self.base_path,
            &self.scanned_files_count,
            shared_frecency,
            self.mode,
        );

        match walk_result {
            Ok(walk) => {
                info!(
                    "Filesystem rescan completed: found {} files",
                    walk.sync.files.len()
                );

                self.sync_data = walk.sync;
                self.cache_budget.reset();

                // Apply git status synchronously for rescan (typically fast).
                if let Ok(Some(git_cache)) = walk.git_handle.join() {
                    let frecency = shared_frecency.read().ok();
                    let frecency_ref = frecency.as_ref().and_then(|f| f.as_ref());
                    let mode = self.mode;
                    let bp = &self.base_path;
                    BACKGROUND_THREAD_POOL.install(|| {
                        self.sync_data.files.par_iter_mut().for_each(|file| {
                            file.git_status = git_cache.lookup_status(&file.absolute_path(bp));
                            if let Some(frecency) = frecency_ref {
                                let _ = file.update_frecency_scores(frecency, bp, mode);
                            }
                        });
                    });
                }

                // Warmup is deferred to the post-rescan bigram rebuild thread
                // (spawned by trigger_full_rescan) which does warmup + bigram
                // in one pass, matching the initial scan's post-scan phase.
            }
            Err(error) => error!(?error, "Failed to scan file system"),
        }

        self.is_scanning.store(false, Ordering::Relaxed);
        Ok(())
    }

    /// Quick way to check if scan is going without acquiring a lock for [Self::get_scan_progress]
    pub fn is_scan_active(&self) -> bool {
        self.is_scanning.load(Ordering::Relaxed)
    }

    /// Return a clone of the scanning flag so callers can poll it without
    /// holding a lock on the picker.
    pub fn scan_signal(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.is_scanning)
    }

    /// Return a clone of the watcher-ready flag so callers can poll it without
    /// holding a lock on the picker.
    pub fn watcher_signal(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.watcher_ready)
    }
}

/// A point-in-time snapshot of the file-scanning progress.
///
/// Returned by [`FilePicker::get_scan_progress`]. Useful for displaying
/// a progress indicator while the initial scan is running.
#[allow(unused)]
#[derive(Debug, Clone)]
pub struct ScanProgress {
    /// Number of files indexed so far.
    pub scanned_files_count: usize,
    /// `true` while the background scan thread is still running.
    pub is_scanning: bool,
    pub is_watcher_ready: bool,
    pub is_warmup_complete: bool,
}

#[allow(clippy::too_many_arguments)]
fn spawn_scan_and_watcher(
    base_path: PathBuf,
    scan_signal: Arc<AtomicBool>,
    watcher_ready: Arc<AtomicBool>,
    synced_files_count: Arc<AtomicUsize>,
    warmup_mmap_cache: bool,
    watch: bool,
    mode: FFFMode,
    shared_picker: SharedPicker,
    shared_frecency: SharedFrecency,
    cancelled: Arc<AtomicBool>,
    post_scan_busy: Arc<AtomicBool>,
) {
    std::thread::spawn(move || {
        // scan_signal is already `true` (set by the caller before spawning)
        // so waiters see "scanning" even before this thread is scheduled.
        info!("Starting initial file scan");

        let git_workdir;

        match walk_filesystem(&base_path, &synced_files_count, &shared_frecency, mode) {
            Ok(walk) => {
                if cancelled.load(Ordering::Acquire) {
                    info!("Walk completed but picker was replaced, discarding results");
                    scan_signal.store(false, Ordering::Relaxed);
                    return;
                }

                info!(
                    "Initial filesystem walk completed: found {} files",
                    walk.sync.files.len()
                );

                git_workdir = walk.sync.git_workdir.clone();
                let git_handle = walk.git_handle;

                // Write files immediately — they are now searchable even
                // before git status or warmup completes.
                let write_result = shared_picker.write().ok().map(|mut guard| {
                    if let Some(ref mut picker) = *guard {
                        picker.sync_data = walk.sync;
                        picker.cache_budget.reset();
                    }
                });

                if write_result.is_none() {
                    error!("Failed to write scan results into picker");
                }

                // Signal scan complete — files are searchable.
                scan_signal.store(false, Ordering::Relaxed);
                info!("Files indexed and searchable");

                // Apply git status (may still be running — this waits for it).
                if !cancelled.load(Ordering::Acquire) {
                    apply_git_status(&shared_picker, &shared_frecency, git_handle, mode);
                }
            }
            Err(e) => {
                error!("Initial scan failed: {:?}", e);
                scan_signal.store(false, Ordering::Relaxed);
                watcher_ready.store(true, Ordering::Release);
                return;
            }
        }

        if watch && !cancelled.load(Ordering::Acquire) {
            let watch_dirs = shared_picker
                .read()
                .ok()
                .and_then(|guard| guard.as_ref().map(|picker| picker.extract_watch_dirs()))
                .unwrap_or_default();

            match BackgroundWatcher::new(
                base_path.clone(),
                git_workdir,
                shared_picker.clone(),
                shared_frecency.clone(),
                mode,
                watch_dirs,
            ) {
                Ok(watcher) => {
                    info!("Background file watcher initialized successfully");

                    if cancelled.load(Ordering::Acquire) {
                        info!("Picker was replaced, dropping orphaned watcher");
                        drop(watcher);
                        watcher_ready.store(true, Ordering::Release);
                        return;
                    }

                    let write_result = shared_picker.write().ok().map(|mut guard| {
                        if let Some(ref mut picker) = *guard {
                            picker.background_watcher = Some(watcher);
                        }
                    });

                    if write_result.is_none() {
                        error!("Failed to store background watcher in picker");
                    }
                }
                Err(e) => {
                    error!("Failed to initialize background file watcher: {:?}", e);
                }
            }
        }

        watcher_ready.store(true, Ordering::Release);

        if warmup_mmap_cache && !cancelled.load(Ordering::Acquire) {
            post_scan_busy.store(true, Ordering::Release);
            let phase_start = std::time::Instant::now();

            // Scale cache limits based on repo size (skip if caller provided an explicit budget).
            if let Ok(mut guard) = shared_picker.write()
                && let Some(ref mut picker) = *guard
                && !picker.has_explicit_cache_budget
            {
                let file_count = picker.sync_data.files().len();
                picker.cache_budget = Arc::new(ContentCacheBudget::new_for_repo(file_count));
                info!(
                    "Cache budget configured for {} files: max_files={}, max_bytes={}",
                    file_count, picker.cache_budget.max_files, picker.cache_budget.max_bytes,
                );
            }

            // SAFETY: The file index Vec is not resized between the initial scan
            // completing and the warmup + bigram phase finishing because
            // `post_scan_busy` prevents concurrent rescans from replacing
            // sync_data while we hold the raw pointer.
            let files_snapshot: Option<(&[FileItem], Arc<ContentCacheBudget>)> =
                if !cancelled.load(Ordering::Acquire) {
                    let guard = shared_picker.read().ok();
                    guard.and_then(|guard| {
                        guard.as_ref().map(|picker| {
                            let files = picker.sync_data.files();
                            let ptr = files.as_ptr();
                            let len = files.len();
                            let budget = Arc::clone(&picker.cache_budget);
                            // SAFETY: post_scan_busy flag blocks trigger_rescan and
                            // background watcher rescans from replacing sync_data,
                            // so the Vec backing this slice stays alive.
                            let static_files: &[FileItem] =
                                unsafe { std::slice::from_raw_parts(ptr, len) };
                            (static_files, budget)
                        })
                    })
                } else {
                    None
                };

            if let Some((files, budget)) = files_snapshot {
                // Warmup: populate mmap caches for top-frecency files.
                if !cancelled.load(Ordering::Acquire) {
                    let warmup_start = std::time::Instant::now();
                    warmup_mmaps(files, &budget, &base_path);
                    info!(
                        "Warmup completed in {:.2}s (cached {} files, {} bytes)",
                        warmup_start.elapsed().as_secs_f64(),
                        budget.cached_count.load(Ordering::Relaxed),
                        budget.cached_bytes.load(Ordering::Relaxed),
                    );
                }

                // Build bigram index — entirely lock-free.
                if !cancelled.load(Ordering::Acquire) {
                    let bigram_start = std::time::Instant::now();
                    info!("Starting bigram index build for {} files...", files.len());
                    let (index, content_binary) = build_bigram_index(files, &budget, &base_path);
                    info!(
                        "Bigram index ready in {:.2}s",
                        bigram_start.elapsed().as_secs_f64(),
                    );

                    if let Ok(mut guard) = shared_picker.write()
                        && let Some(ref mut picker) = *guard
                    {
                        for &idx in &content_binary {
                            if let Some(file) = picker.sync_data.get_file_mut(idx) {
                                file.set_binary(true);
                            }
                        }

                        let base_count = picker.sync_data.base_count;
                        picker.sync_data.bigram_index = Some(Arc::new(index));
                        picker.sync_data.bigram_overlay = Some(Arc::new(parking_lot::RwLock::new(
                            BigramOverlay::new(base_count),
                        )));
                    }
                }
            }

            post_scan_busy.store(false, Ordering::Release);

            info!(
                "Post-scan warmup + bigram total: {:.2}s",
                phase_start.elapsed().as_secs_f64(),
            );
        }

        // the debouncer keeps running in its own thread
    });
}

/// Pre-populate mmap caches for the most valuable files so the first grep
/// search doesn't pay the mmap creation + page fault cost.
///
/// All files are collected once, then an O(n) `select_nth_unstable_by`
/// partitions the top [`MAX_CACHED_CONTENT_FILES`] highest-frecency eligible
/// files to the front (binary / empty files are pushed to the end by the
/// comparator). The selected prefix is warmed in parallel via rayon.
///
/// Files beyond the budget are still available via temporary mmaps on first
/// grep access, so correctness is unaffected.
#[tracing::instrument(skip(files), name = "warmup_mmaps", level = Level::DEBUG)]
pub fn warmup_mmaps(files: &[FileItem], budget: &ContentCacheBudget, base_path: &Path) {
    let max_files = budget.max_files;
    let max_bytes = budget.max_bytes;
    let max_file_size = budget.max_file_size;

    // Single collect — no pre-filter. The comparator in select_nth pushes
    // ineligible files (binary, empty) to the tail automatically.
    let mut all: Vec<&FileItem> = files.iter().collect();

    // O(n) partial sort: top max_files eligible-by-frecency files land in
    // all[..max_files]. Ineligible files compare as "lowest priority" so
    // they naturally sink past the partition boundary.
    if all.len() > max_files {
        all.select_nth_unstable_by(max_files, |a, b| {
            let a_ok = !a.is_binary() && a.size > 0;
            let b_ok = !b.is_binary() && b.size > 0;
            match (a_ok, b_ok) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                (false, false) => std::cmp::Ordering::Equal,
                (true, true) => b.total_frecency_score().cmp(&a.total_frecency_score()),
            }
        });
    }

    let to_warm = &all[..all.len().min(max_files)];

    let warmed_bytes = AtomicU64::new(0);
    let budget_exhausted = AtomicBool::new(false);

    BACKGROUND_THREAD_POOL.install(|| {
        to_warm.par_iter().for_each(|file| {
            if budget_exhausted.load(Ordering::Relaxed) {
                return;
            }

            if file.is_binary() || file.size == 0 || file.size > max_file_size {
                return;
            }

            // Byte budget.
            let prev_bytes = warmed_bytes.fetch_add(file.size, Ordering::Relaxed);
            if prev_bytes + file.size > max_bytes {
                budget_exhausted.store(true, Ordering::Relaxed);
                return;
            }

            if let Some(content) = file.get_content(base_path, budget) {
                let _ = std::hint::black_box(content.first());
            }
        });
    });
}

/// Max bytes of file content scanned for bigram indexing. After this many
/// bytes the ~4900 possible printable-ASCII bigrams are effectively saturated,
/// so reading further adds no new information to the index.
pub const BIGRAM_CONTENT_CAP: usize = 64 * 1024;

pub fn build_bigram_index(
    files: &[FileItem],
    budget: &ContentCacheBudget,
    base_path: &Path,
) -> (BigramFilter, Vec<usize>) {
    let start = std::time::Instant::now();
    info!("Building bigram index for {} files...", files.len());
    let builder = BigramIndexBuilder::new(files.len());
    let skip_builder = BigramIndexBuilder::new(files.len());
    let max_file_size = budget.max_file_size;

    // Collect indices of files that passed the extension heuristic but are
    // actually binary (contain NUL bytes). These are marked `is_binary = true`
    // on the real file list after the build, so grep never has to re-check.
    let content_binary: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

    BACKGROUND_THREAD_POOL.install(|| {
        files.par_iter().enumerate().for_each(|(i, file)| {
            if file.is_binary() || file.size == 0 || file.size > max_file_size {
                return;
            }
            // Use cached content if available (no extra memory).
            // For uncached files, read from disk — heap memory is freed on drop.
            let data: Option<&[u8]>;
            let owned;
            if let Some(cached) = file.get_content(base_path, budget) {
                if detect_binary_content(cached) {
                    content_binary.lock().unwrap().push(i);
                    return;
                }
                data = Some(cached);
                owned = None;
            } else if let Ok(read_data) = std::fs::read(file.absolute_path(base_path)) {
                if detect_binary_content(&read_data) {
                    content_binary.lock().unwrap().push(i);
                    return;
                }
                data = None;
                owned = Some(read_data);
            } else {
                return;
            }

            let content = data.unwrap_or_else(|| owned.as_ref().unwrap());
            let capped = &content[..content.len().min(BIGRAM_CONTENT_CAP)];
            builder.add_file_content(&skip_builder, i, capped);
        });
    });

    let cols = builder.columns_used();
    let mut index = builder.compress(None);

    // Skip bigrams are supplementary — the consecutive index does the heavy
    // lifting. Rare skip columns (< 12% of files) add virtually no filtering
    // on either homogeneous (kernel) or polyglot (monorepo) codebases, but
    // cost ~25-30% of total index memory. Using a higher sparse cutoff for
    // the skip index drops these dead-weight columns with negligible loss.
    let skip_index = skip_builder.compress(Some(12));
    index.set_skip_index(skip_index);

    // The builders' flat buffers were freed by compress() above (single
    // deallocation each). Hint the allocator to return pages from other
    // per-thread allocations (file reads, sort buffers) during the build.
    hint_allocator_collect();

    info!(
        "Bigram index built in {:.2}s — {} dense columns for {} files",
        start.elapsed().as_secs_f64(),
        cols,
        files.len(),
    );

    let binary_indices = content_binary.into_inner().unwrap();
    if !binary_indices.is_empty() {
        info!(
            "Bigram build detected {} content-binary files (not caught by extension)",
            binary_indices.len(),
        );
    }

    (index, binary_indices)
}

// pub for benchmarks
pub fn scan_files(base_path: &Path) -> Vec<FileItem> {
    use ignore::{WalkBuilder, WalkState};

    let git_workdir = Repository::discover(base_path)
        .ok()
        .and_then(|repo| repo.workdir().map(Path::to_path_buf));
    let is_git_repo = git_workdir.is_some();

    let mut walk_builder = WalkBuilder::new(base_path);
    walk_builder
        .hidden(!is_git_repo)
        .git_ignore(true)
        .git_exclude(true)
        .git_global(true)
        .ignore(true)
        .follow_links(false);

    if !is_git_repo && let Some(overrides) = non_git_repo_overrides(base_path) {
        walk_builder.overrides(overrides);
    }

    let walker = walk_builder.build_parallel();
    let files = parking_lot::Mutex::new(Vec::new());

    walker.run(|| {
        let files = &files;
        let base_path = base_path.to_path_buf();

        Box::new(move |result| {
            let Ok(entry) = result else {
                return WalkState::Continue;
            };

            if entry.file_type().is_some_and(|ft| ft.is_file()) {
                let path = entry.path();

                if is_git_file(path) {
                    return WalkState::Continue;
                }

                if !is_git_repo && is_known_binary_extension(path) {
                    return WalkState::Continue;
                }

                let metadata = entry.metadata().ok();
                let file_item = FileItem::new_with_metadata(
                    path.to_path_buf(),
                    &base_path,
                    None,
                    metadata.as_ref(),
                );

                files.lock().push(file_item);
            }
            WalkState::Continue
        })
    });

    let pairs = files.into_inner();
    // Leak the path strings so FileItem path_ptrs remain valid for the test.
    // Tests are short-lived processes so this is acceptable.
    let mut files: Vec<FileItem> = Vec::with_capacity(pairs.len());
    for (file, path_str) in pairs {
        std::mem::forget(path_str); // keep heap data alive for path_ptr
        files.push(file);
    }
    files.sort_unstable_by(|a, b| a.file_name().cmp(b.file_name()));
    files
}

/// Result of the fast walk phase — files are searchable immediately,
/// git status arrives later via the join handle.
struct WalkResult {
    sync: FileSync,
    git_handle: std::thread::JoinHandle<Option<GitStatusCache>>,
}

/// Build the SIMD-chunked directory arena from the sorted dir table.
///
/// Directory paths are deduplicated — each unique dir is stored as a sequence
/// of 16-byte aligned chunks, zero-padded. Repoints each FileItem's `dir_ptr`
/// from the temporary flat dir arena into the permanent chunk arena.
///
/// Filenames are already in the permanent `filename_arena` from the walk phase.
///
/// SAFETY: The chunk arena is pre-allocated to exact capacity and never reallocated.
fn build_dir_chunk_arena(
    files: &mut [FileItem],
    dirs: &[DirItem],
) -> Vec<crate::simd_path::SimdChunk> {
    use crate::simd_path::SimdChunk;

    // ── Phase 1: Calculate total chunks needed ──
    let mut total_dir_chunks = 0usize;
    let mut dir_meta: Vec<(usize, u16)> = Vec::with_capacity(dirs.len());

    for dir in dirs {
        let rel = dir.relative_path();
        let dir_bytes_len = if rel.is_empty() { 0 } else { rel.len() + 1 };
        let n_chunks = if dir_bytes_len == 0 {
            0
        } else {
            (dir_bytes_len + 15) / 16
        };
        dir_meta.push((total_dir_chunks * 16, dir_bytes_len as u16));
        total_dir_chunks += n_chunks;
    }

    let mut chunk_arena: Vec<SimdChunk> = Vec::with_capacity(total_dir_chunks);

    // ── Phase 2: Fill directory chunk arena ──
    for dir in dirs {
        let rel = dir.relative_path();
        if rel.is_empty() {
            continue;
        }
        let bytes = rel.as_bytes();
        let dir_bytes_len = bytes.len() + 1;
        let n_chunks = (dir_bytes_len + 15) / 16;
        for i in 0..n_chunks {
            let mut chunk = SimdChunk::default();
            let start = i * 16;
            let end = (start + 16).min(dir_bytes_len);
            let rel_end = end.min(bytes.len());
            if start < bytes.len() {
                chunk.as_bytes_mut()[..rel_end - start].copy_from_slice(&bytes[start..rel_end]);
            }
            if start <= bytes.len() && bytes.len() < end {
                chunk.as_bytes_mut()[bytes.len() - start] = b'/';
            }
            chunk_arena.push(chunk);
        }
    }

    debug_assert_eq!(chunk_arena.len(), total_dir_chunks);
    debug_assert_eq!(chunk_arena.capacity(), total_dir_chunks);

    // ── Phase 3: Repoint dir_ptr for each FileItem ──
    let chunk_base_ptr = chunk_arena.as_ptr() as *const u8;

    for file in files.iter_mut() {
        let dir_idx = file.parent_dir_index() as usize;
        let (dir_byte_offset, _dir_byte_len) = dir_meta[dir_idx];
        let dir_ptr = unsafe { chunk_base_ptr.add(dir_byte_offset) };
        unsafe { file.repoint_dir(dir_ptr) };
    }

    chunk_arena
}

/// Build the directory table from a sorted file list. For each file, extracts
/// the parent directory path. Produces a sorted `Vec<DirItem>` with per-dir
/// stats, and assigns `parent_dir` indices back to each `FileItem`.
///
/// Two-phase approach for fast sorting:
///
/// **Phase 1** (`assign_parent_dirs`): Runs BEFORE file sort. Extracts unique
/// parent dirs into a HashMap, sorts them, assigns `parent_dir: u32` index to
/// each file. O(n) for extraction + O(d log d) for dir sort + O(n) for assignment.
///
/// **Phase 2** (`compute_dir_stats`): Runs AFTER file sort. Single O(n) pass over
/// sorted files to compute per-dir stats (file_count, max_frecency, last_modified).
///
/// This enables sorting files by `(parent_dir, filename)` instead of full path
/// string — a single u32 comparison resolves ~99.7% of sort comparisons
/// (files in different directories), with only the rare same-directory case
/// requiring a short filename comparison.

/// Phase 1: Extract unique parent directories, sort them, assign indices to files.
/// Returns a `Vec<DirItem>` with paths set but stats zeroed (filled in Phase 2).
fn assign_parent_dirs(files: &mut [FileItem], base_path: &Path) -> Vec<DirItem> {
    if files.is_empty() {
        return Vec::new();
    }

    let base_str = base_path.as_os_str().to_string_lossy();
    let base_len = base_str.len();

    // Pass 1: collect unique parent directory paths (absolute)
    let mut dir_map: ahash::HashMap<String, u32> = ahash::HashMap::default();
    dir_map.reserve(files.len() / 4);
    let mut dir_entries: Vec<(String, u16)> = Vec::with_capacity(files.len() / 4);

    for file in files.iter() {
        let rel = file.relative_path();
        // Reconstruct absolute path for directory lookup
        let abs_path = if rel.is_empty() {
            base_str.to_string()
        } else {
            format!("{}/{}", base_str, rel)
        };
        let parent_end = abs_path.rfind('/').unwrap_or(base_len);
        let parent = &abs_path[..parent_end];

        if !dir_map.contains_key(parent) {
            let relative_start = if parent_end > base_len {
                (base_len + 1) as u16
            } else {
                parent_end as u16
            };
            dir_map.insert(parent.to_string(), dir_entries.len() as u32);
            dir_entries.push((parent.to_string(), relative_start));
        }
    }

    // Sort directory entries by path to establish stable sort order
    dir_entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    // Rebuild map with sorted indices
    let sorted_map: ahash::HashMap<&str, u32> = dir_entries
        .iter()
        .enumerate()
        .map(|(i, (path, _))| (path.as_str(), i as u32))
        .collect();

    // Pass 2: assign parent_dir index to each file
    for file in files.iter_mut() {
        let rel = file.relative_path();
        let abs_path = if rel.is_empty() {
            base_str.to_string()
        } else {
            format!("{}/{}", base_str, rel)
        };
        let parent_end = abs_path.rfind('/').unwrap_or(base_len);
        let parent = &abs_path[..parent_end];
        file.set_parent_dir(*sorted_map.get(parent).unwrap());
    }

    // Create DirItem entries (stats zeroed, filled by compute_dir_stats)
    let dirs: Vec<DirItem> = dir_entries
        .into_iter()
        .map(|(path, relative_start)| DirItem::new(path, relative_start))
        .collect();

    dirs
}

/// Phase 2: Compute per-directory stats from the sorted file list.
/// Files must already be sorted by (parent_dir, filename) and have parent_dir assigned.
fn compute_dir_stats(files: &[FileItem], dirs: &mut [DirItem]) {
    for file in files {
        let idx = file.parent_dir_index() as usize;
        if idx < dirs.len() {
            let dir = &mut dirs[idx];
            dir.file_count += 1;
            dir.last_modified = dir.last_modified.max(file.modified);
            dir.max_child_frecency = dir
                .max_child_frecency
                .max(file.access_frecency_score + file.modification_frecency_score);
        }
    }
}

/// Returns files immediately (searchable) and a handle to the in-progress
/// git status computation. This avoids blocking on `git status` which can
/// take 10+ seconds on very large repos (e.g. chromium).
fn walk_filesystem(
    base_path: &Path,
    synced_files_count: &Arc<AtomicUsize>,
    shared_frecency: &SharedFrecency,
    mode: FFFMode,
) -> Result<WalkResult, Error> {
    use ignore::{WalkBuilder, WalkState};

    let scan_start = std::time::Instant::now();
    info!("SCAN: Starting filesystem walk and git status (async)");

    // Discover git root (fast — just walks up looking for .git/)
    let git_workdir = Repository::discover(base_path)
        .ok()
        .and_then(|repo| repo.workdir().map(Path::to_path_buf));

    if let Some(ref git_dir) = git_workdir {
        debug!("Git repository found at: {}", git_dir.display());
    } else {
        debug!("No git repository found for path: {}", base_path.display());
    }

    // Spawn git status on a detached thread — we won't wait for it here.
    let git_workdir_for_status = git_workdir.clone();
    let git_handle = std::thread::spawn(move || {
        GitStatusCache::read_git_status(
            git_workdir_for_status.as_deref(),
            StatusOptions::new()
                .include_untracked(true)
                .recurse_untracked_dirs(true)
                .exclude_submodules(true),
        )
    });

    // Walk files (the fast part, typically 2-3s even on huge repos).
    let is_git_repo = git_workdir.is_some();
    let bg_threads = BACKGROUND_THREAD_POOL.current_num_threads();
    let mut walk_builder = WalkBuilder::new(base_path);
    walk_builder
        // this is a very important guard for the user opening ~/ or other root non-git dir
        .hidden(!is_git_repo)
        .git_ignore(true)
        .git_exclude(true)
        .git_global(true)
        .ignore(true)
        .follow_links(false)
        .threads(bg_threads);

    if !is_git_repo && let Some(overrides) = non_git_repo_overrides(base_path) {
        walk_builder.overrides(overrides);
    }

    let walker = walk_builder.build_parallel();

    let walker_start = std::time::Instant::now();
    debug!("SCAN: Starting file walker");

    let files = parking_lot::Mutex::new(Vec::new());
    // Flat arenas populated during the walk — no per-file String allocation.
    // dir_arena is temporary (replaced by SIMD chunk arena after sort).
    // filename_arena is permanent (FileItem::filename_ptr points here forever).
    // dir_dedup: maps dir_part string to (offset, len) in dir_arena.
    // Using &str keys would require borrowing from the arena we're mutating,
    // so we store small owned keys. Only ~36K unique dirs on chromium.
    let dir_arena = parking_lot::Mutex::new(Vec::<u8>::new());
    let dir_dedup = parking_lot::Mutex::new(ahash::AHashMap::<Box<[u8]>, (u32, u16)>::new());
    let filename_arena = parking_lot::Mutex::new(Vec::<u8>::new());

    walker.run(|| {
        let files = &files;
        let dir_arena = &dir_arena;
        let dir_dedup = &dir_dedup;
        let filename_arena = &filename_arena;
        let counter = Arc::clone(synced_files_count);
        let base_path = base_path.to_path_buf();

        Box::new(move |result| {
            let Ok(entry) = result else {
                return WalkState::Continue;
            };

            if entry.file_type().is_some_and(|ft| ft.is_file()) {
                let path = entry.path();

                if is_git_file(path) {
                    return WalkState::Continue;
                }

                if !is_git_repo && is_known_binary_extension(path) {
                    return WalkState::Continue;
                }

                let metadata = entry.metadata().ok();
                // Lock all three under one scope to avoid interleaving
                let mut da = dir_arena.lock();
                let mut dd = dir_dedup.lock();
                let mut fa = filename_arena.lock();
                let file_item = FileItem::new_into_arenas(
                    path.to_path_buf(),
                    &base_path,
                    None,
                    metadata.as_ref(),
                    &mut da,
                    &mut dd,
                    &mut fa,
                );
                drop(da);
                drop(dd);
                drop(fa);

                files.lock().push(file_item);
                counter.fetch_add(1, Ordering::Relaxed);
            }
            WalkState::Continue
        })
    });

    let mut files = files.into_inner();
    let temp_dir_arena = dir_arena.into_inner();
    let filename_arena = filename_arena.into_inner();

    // Convert offset-based pointers to real pointers into the final arenas.
    // During the walk, dir_ptr/filename_ptr stored byte offsets (as usize cast
    // to *const u8) because the arenas could reallocate.
    let dir_base = temp_dir_arena.as_ptr();
    let fname_base = filename_arena.as_ptr();
    for file in files.iter_mut() {
        let dir_offset = file.dir_ptr_raw() as usize;
        let fname_offset = file.filename_ptr_raw() as usize;
        unsafe {
            file.repoint_dir(dir_base.add(dir_offset));
            file.repoint_filename(fname_base.add(fname_offset));
        }
    }

    info!(
        "SCAN: File walking completed in {:?} for {} files",
        walker_start.elapsed(),
        files.len(),
    );

    // Apply frecency scores (access-based only — git status not yet available).
    let frecency = shared_frecency
        .read()
        .map_err(|_| Error::AcquireFrecencyLock)?;
    if let Some(frecency) = frecency.as_ref() {
        BACKGROUND_THREAD_POOL.install(|| {
            files.par_iter_mut().for_each(|file| {
                let _ = file.update_frecency_scores(frecency, base_path, mode);
            });
        });
    }
    drop(frecency);

    // Phase 1: Extract unique dirs, sort them, assign parent_dir to each file.
    // This enables fast sorting by (parent_dir, filename) instead of full path string.
    let mut dirs = assign_parent_dirs(&mut files, base_path);

    // Sort files by (parent_dir, filename). The u32 parent_dir comparison
    // resolves immediately for files in different directories (~99.7% of
    // comparisons). Only same-directory files need the short filename comparison.
    BACKGROUND_THREAD_POOL.install(|| {
        files.par_sort_unstable_by(|a, b| {
            a.parent_dir_index()
                .cmp(&b.parent_dir_index())
                .then_with(|| a.file_name().cmp(b.file_name()))
        });
    });

    // Phase 2: Compute per-directory stats from the now-sorted files.
    compute_dir_stats(&files, &mut dirs);

    // Build SIMD-chunked directory arena (deduplicated).
    // Only repoints dir_ptr — filename_ptr already points into the permanent filename_arena.
    let dir_chunk_arena = build_dir_chunk_arena(&mut files, &dirs);

    // NOW safe to drop the temporary flat dir arena — dir_ptrs point to chunk arena.
    drop(temp_dir_arena);

    // Ask the allocator to return freed pages to the OS.
    hint_allocator_collect();

    let file_item_size = std::mem::size_of::<FileItem>();
    let files_vec_bytes = files.len() * file_item_size;
    let dir_table_bytes = dirs.len() * std::mem::size_of::<DirItem>()
        + dirs.iter().map(|d| d.path_str().len()).sum::<usize>();
    let arena_bytes = dir_chunk_arena.len() * 16 + filename_arena.len();

    let total_time = scan_start.elapsed();
    info!(
        "SCAN: Walk completed in {:?} ({} files, {} dirs, \
         arena={:.2}MB, files_vec={:.2}MB, dirs={:.2}MB, FileItem={}B)",
        total_time,
        files.len(),
        dirs.len(),
        arena_bytes as f64 / 1_048_576.0,
        files_vec_bytes as f64 / 1_048_576.0,
        dir_table_bytes as f64 / 1_048_576.0,
        file_item_size,
    );

    let base_count = files.len();
    Ok(WalkResult {
        sync: FileSync {
            files,
            base_count,
            dirs,
            dir_chunk_arena,
            filename_arena,
            overflow_arena: Vec::new(),
            git_workdir,
            bigram_index: None,
            bigram_overlay: None,
            path_bigram_index: None,
        },
        git_handle,
    })
}

/// Phase 2: apply git status to already-indexed files and recalculate
/// frecency scores that depend on it.
fn apply_git_status(
    shared_picker: &SharedPicker,
    shared_frecency: &SharedFrecency,
    git_handle: std::thread::JoinHandle<Option<GitStatusCache>>,
    mode: FFFMode,
) {
    let join_start = std::time::Instant::now();
    let git_cache = match git_handle.join() {
        Ok(cache) => cache,
        Err(_) => {
            error!("Git status thread panicked");
            return;
        }
    };
    info!("SCAN: Git status ready in {:?}", join_start.elapsed());

    let Some(git_cache) = git_cache else { return };

    if let Ok(mut guard) = shared_picker.write()
        && let Some(ref mut picker) = *guard
    {
        let frecency = shared_frecency.read().ok();
        let frecency_ref = frecency.as_ref().and_then(|f| f.as_ref());

        BACKGROUND_THREAD_POOL.install(|| {
            let bp = &picker.base_path;
            picker.sync_data.files.par_iter_mut().for_each(|file| {
                file.git_status = git_cache.lookup_status(&file.absolute_path(bp));
                if let Some(frecency) = frecency_ref {
                    let _ = file.update_frecency_scores(frecency, bp, mode);
                }
            });
        });

        info!(
            "SCAN: Applied git status to {} files ({} dirty)",
            picker.sync_data.files.len(),
            git_cache.statuses_len(),
        );
    }
}

#[inline]
fn is_git_file(path: &Path) -> bool {
    path.to_str().is_some_and(|path| {
        if cfg!(target_family = "windows") {
            path.contains("\\.git\\")
        } else {
            path.contains("/.git/")
        }
    })
}

/// Fast extension-based binary detection. Avoids opening files during scan.
/// Covers the vast majority of binary files in typical repositories.
#[inline]
fn is_known_binary_extension(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    matches!(
        ext,
        // Images
        "png" | "jpg" | "jpeg" | "gif" | "bmp" | "ico" | "webp" | "tiff" | "tif" | "avif" |
        "heic" | "psd" | "icns" | "cur" | "raw" | "cr2" | "nef" | "dng" |
        // Video/Audio
        "mp4" | "avi" | "mov" | "wmv" | "mkv" | "mp3" | "wav" | "flac" | "ogg" | "m4a" |
        "aac" | "webm" | "flv" | "mpg" | "mpeg" | "wma" | "opus" |
        // Compressed/Archives
        "zip" | "tar" | "gz" | "bz2" | "xz" | "7z" | "rar" | "zst" | "lz4" | "lzma" |
        "cab" | "cpio" |
        // Packages/Installers
        "deb" | "rpm" | "apk" | "dmg" | "msi" | "iso" | "nupkg" | "whl" | "egg" |
        "snap" | "appimage" | "flatpak" |
        // Executables/Libraries
        "exe" | "dll" | "so" | "dylib" | "o" | "a" | "lib" | "bin" | "elf" |
        // Documents
        "pdf" | "doc" | "docx" | "xls" | "xlsx" | "ppt" | "pptx" |
        // Databases
        "db" | "sqlite" | "sqlite3" | "mdb" |
        // Fonts
        "ttf" | "otf" | "woff" | "woff2" | "eot" |
        // Compiled/Runtime
        "class" | "pyc" | "pyo" | "wasm" | "dex" | "jar" | "war" |
        // ML/Data Science
        "npy" | "npz" | "pkl" | "pickle" | "h5" | "hdf5" | "pt" | "pth" | "onnx" |
        "safetensors" | "tfrecord" |
        // 3D/Game
        "glb" | "fbx" | "blend" |
        // Data/serialized
        "parquet" | "arrow" | "pb" |
        // IDE/OS metadata
        "DS_Store" | "suo"
    )
}

/// Detect binary content by checking for NUL bytes in the first 512 bytes.
/// Called lazily when file content is first loaded, not during initial scan.
#[inline]
pub(crate) fn detect_binary_content(content: &[u8]) -> bool {
    let check_len = content.len().min(512);
    content[..check_len].contains(&0)
}

/// Ask the global allocator to return freed pages to the OS.
/// Enabled via the `mimalloc-collect` feature (set by fff-nvim).
/// No-op when the feature is off (tests, system allocator).
fn hint_allocator_collect() {
    #[cfg(feature = "mimalloc-collect")]
    {
        // Collect BACKGROUND_THREAD_POOL workers — that's where the bigram
        // builder allocated memory. `rayon::broadcast` would target the global
        // pool, which is the wrong set of threads.
        BACKGROUND_THREAD_POOL.broadcast(|_| unsafe { libmimalloc_sys::mi_collect(true) });

        // Main thread too.
        unsafe { libmimalloc_sys::mi_collect(true) };
    }
}
