use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;

use crate::constraints::Constrainable;
use crate::query_tracker::QueryMatchEntry;
use fff_query_parser::{FFFQuery, FuzzyQuery, Location};
use neo_frizbee::MatchableSegmented;

/// Cached file contents — mmap on Unix, heap buffer on Windows.
///
/// On Windows, memory-mapped files hold the file handle open and prevent
/// editors from saving (writing/replacing) those files. Reading into a
/// `Vec<u8>` releases the handle immediately after the read completes.
///
/// The `Buffer` variant is also used on Unix for temporary (uncached) reads
/// where the mmap/munmap syscall overhead exceeds the cost of a heap copy.
#[derive(Debug)]
#[allow(dead_code)] // variants are conditionally used per platform
pub enum FileContent {
    #[cfg(not(target_os = "windows"))]
    Mmap(memmap2::Mmap),
    Buffer(Vec<u8>),
}

impl std::ops::Deref for FileContent {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            #[cfg(not(target_os = "windows"))]
            FileContent::Mmap(m) => m,
            FileContent::Buffer(b) => b,
        }
    }
}

pub struct FileItemFlags;

impl FileItemFlags {
    pub const BINARY: u8 = 1 << 0;
    /// Tombstone — file was deleted but index slot is preserved so
    /// bigram indices for other files stay valid.
    pub const DELETED: u8 = 1 << 1;
}

/// A directory entry with aggregated metadata from its child files.
/// Stored in a sorted `Vec<DirItem>` (the "dir table") inside `FileSync`,
/// giving O(log n) lookup by path and enabling directory picker mode.
#[derive(Debug, Clone)]
pub struct DirItem {
    /// Absolute path of the directory (with trailing separator removed).
    path: String,
    /// Byte offset where the relative path begins (mirrors `FileItem::relative_start`).
    relative_start: u16,
    /// Number of direct child files in this directory.
    pub file_count: u32,
    /// Highest frecency score among direct child files.
    /// Useful for ranking directories in directory-mode search.
    pub max_child_frecency: i16,
    /// Modification time of the most recently modified direct child.
    pub last_modified: u64,
}

impl DirItem {
    pub fn new(path: String, relative_start: u16) -> Self {
        Self {
            path,
            relative_start,
            file_count: 0,
            max_child_frecency: 0,
            last_modified: 0,
        }
    }

    /// The full absolute path as a string slice.
    #[inline]
    pub fn path_str(&self) -> &str {
        &self.path
    }

    /// The full absolute path as a `&Path`.
    #[inline]
    pub fn as_path(&self) -> &Path {
        Path::new(&self.path)
    }

    /// The relative path from the base directory (without trailing separator).
    /// For the base directory itself, returns "".
    #[inline]
    pub fn relative_path(&self) -> &str {
        &self.path[self.relative_start as usize..]
    }
}

impl neo_frizbee::Matchable for DirItem {
    #[inline]
    fn match_str(&self) -> Option<&str> {
        let rel = self.relative_path();
        if rel.is_empty() {
            None
        } else {
            Some(rel)
        }
    }
}

/// A single indexed file with metadata, frecency scores, and lazy content cache.
///
/// File contents are initialized lazily on the first grep access and cached for
/// subsequent searches. On Unix, uses mmap backed by the kernel page cache. On
/// Windows, reads into a heap buffer to avoid holding file handles open.
///
/// Thread-safety: `OnceLock` provides lock-free reads after initialization.
/// Each file is only searched by one rayon worker at a time via `par_iter`.
///
/// Path storage — split `[dir, filename]` segments backed by SIMD-chunked arenas:
///
/// - `dir_ptr` + `dir_len`: directory relative path (e.g. `src/components/`),
///   shared across all files in the same directory via the SIMD chunk arena
///   (16-byte aligned, deduplicated).
/// - `filename_ptr` + `filename_len`: the filename (e.g. `Button.tsx`),
///   stored in a packed filename arena.
///
/// `MatchableSegmented` provides zero-copy `[dir, filename]` segments for SIMD matching.
/// Full path reconstruction (`dir ++ filename`) happens only on cold output paths.
#[derive(Debug)]
pub struct FileItem {
    /// File size in bytes
    pub size: u64,
    /// Modification time in UNIX timestamp
    pub modified: u64,
    /// Frecency access score
    pub access_frecency_score: i16,
    /// Frecency modification score
    pub modification_frecency_score: i16,
    /// The file's git status
    pub git_status: Option<git2::Status>,

    /// Raw pointer to the directory portion of the relative path in the
    /// SIMD chunk arena (16-byte aligned). Empty for root-level files.
    dir_ptr: *const u8,
    /// Length of the directory portion in bytes (e.g. `src/lib/` = 8).
    dir_len: u16,
    /// Raw pointer to the filename in the filename arena.
    filename_ptr: *const u8,
    /// Length of the filename in bytes.
    filename_len: u16,
    /// Index into the dir table (`FileSync::dirs`).
    parent_dir: u32,
    /// Packed boolean flags — see `FileItemFlags`.
    flags: u8,
    /// Lazily-initialized file contents for grep.
    content: OnceLock<FileContent>,
}

// SAFETY: All raw pointers point into immutable arenas owned by FileSync.
// Base files: dir_ptr → simd chunk arena, filename_ptr → filename arena.
// Overflow files: both ptrs → per-file Box<[u8]> in overflow arena.
unsafe impl Send for FileItem {}
unsafe impl Sync for FileItem {}

impl Clone for FileItem {
    fn clone(&self) -> Self {
        Self {
            dir_ptr: self.dir_ptr,
            dir_len: self.dir_len,
            filename_ptr: self.filename_ptr,
            filename_len: self.filename_len,
            parent_dir: self.parent_dir,
            size: self.size,
            modified: self.modified,
            access_frecency_score: self.access_frecency_score,
            modification_frecency_score: self.modification_frecency_score,
            git_status: self.git_status,
            flags: self.flags,
            content: OnceLock::new(),
        }
    }
}

impl FileItem {
    /// Create a new `FileItem`. Initially `dir_ptr` / `filename_ptr` point
    /// into the `abs_path` String.
    ///
    /// Caller must ensure the backing storage outlives this FileItem until
    /// `repoint_dir` / `repoint_filename` is called to repoint into the
    /// packed arenas.
    pub fn new_raw(
        abs_path: &str,
        relative_start: u16,
        filename_start: u16,
        size: u64,
        modified: u64,
        git_status: Option<git2::Status>,
        is_binary: bool,
    ) -> Self {
        let mut flags = 0u8;
        if is_binary {
            flags |= FileItemFlags::BINARY;
        }

        let rel_start = relative_start as usize;
        let fname_start = filename_start as usize;

        // dir portion: abs_path[rel_start..fname_start] (includes trailing /)
        let dir_ptr = unsafe { abs_path.as_ptr().add(rel_start) };
        let dir_len = (fname_start - rel_start) as u16;

        // filename portion: abs_path[fname_start..]
        let filename_ptr = unsafe { abs_path.as_ptr().add(fname_start) };
        let filename_len = (abs_path.len() - fname_start) as u16;

        Self {
            dir_ptr,
            dir_len,
            filename_ptr,
            filename_len,
            parent_dir: u32::MAX,
            size,
            modified,
            access_frecency_score: 0,
            modification_frecency_score: 0,
            git_status,
            flags,
            content: OnceLock::new(),
        }
    }

    /// Create a FileItem from pre-computed pointers into arenas.
    /// Used by the walk phase to avoid per-file String allocations.
    #[inline]
    pub(crate) fn from_arena_ptrs(
        dir_ptr: *const u8,
        dir_len: u16,
        filename_ptr: *const u8,
        filename_len: u16,
        size: u64,
        modified: u64,
        git_status: Option<git2::Status>,
        is_binary: bool,
    ) -> Self {
        let mut flags = 0u8;
        if is_binary {
            flags |= FileItemFlags::BINARY;
        }
        Self {
            dir_ptr,
            dir_len,
            filename_ptr,
            filename_len,
            parent_dir: u32::MAX,
            size,
            modified,
            access_frecency_score: 0,
            modification_frecency_score: 0,
            git_status,
            flags,
            content: OnceLock::new(),
        }
    }

    /// Repoint dir_ptr into the SIMD chunk arena.
    ///
    /// # Safety
    /// `ptr` must point to valid UTF-8 of at least `self.dir_len` bytes
    /// in memory that outlives this FileItem.
    #[inline]
    pub unsafe fn repoint_dir(&mut self, ptr: *const u8) {
        self.dir_ptr = ptr;
    }

    /// Repoint filename_ptr into the filename arena.
    ///
    /// # Safety
    /// `ptr` must point to valid UTF-8 of at least `self.filename_len` bytes
    /// in memory that outlives this FileItem.
    #[inline]
    pub unsafe fn repoint_filename(&mut self, ptr: *const u8) {
        self.filename_ptr = ptr;
    }

    /// Raw dir_ptr value (may be an offset during the walk phase).
    #[inline]
    pub(crate) fn dir_ptr_raw(&self) -> *const u8 {
        self.dir_ptr
    }

    /// Raw filename_ptr value (may be an offset during the walk phase).
    #[inline]
    pub(crate) fn filename_ptr_raw(&self) -> *const u8 {
        self.filename_ptr
    }

    /// Index into the dir table for this file's parent directory.
    #[inline]
    pub fn parent_dir_index(&self) -> u32 {
        self.parent_dir
    }

    /// Set the parent directory index.
    #[inline]
    pub fn set_parent_dir(&mut self, idx: u32) {
        self.parent_dir = idx;
    }

    /// The directory portion of the relative path. Zero-cost slice.
    ///
    /// For `src/components/Button.tsx` returns `"src/components/"`.
    /// For root-level files returns `""`.
    #[inline]
    pub fn dir_str(&self) -> &str {
        if self.dir_len == 0 {
            return "";
        }
        unsafe {
            let slice = std::slice::from_raw_parts(self.dir_ptr, self.dir_len as usize);
            std::str::from_utf8_unchecked(slice)
        }
    }

    /// The full relative path. **Allocates** — cold-path only.
    ///
    /// For the hot matching path, use `MatchableSegmented::match_segments()`
    /// which returns zero-copy `[dir, filename]` segments.
    #[inline]
    pub fn relative_path(&self) -> String {
        let dir = self.dir_str();
        let filename = self.file_name();
        let mut s = String::with_capacity(dir.len() + filename.len());
        s.push_str(dir);
        s.push_str(filename);
        s
    }

    /// Check if the relative path equals `other` without allocating.
    #[inline]
    pub fn relative_path_eq(&self, other: &str) -> bool {
        let dir_len = self.dir_len as usize;
        let fname_len = self.filename_len as usize;
        other.len() == dir_len + fname_len
            && other[..dir_len].as_bytes() == self.dir_bytes()
            && other[dir_len..].as_bytes() == self.filename_bytes()
    }

    /// Check if the relative path ends with `suffix` without allocating.
    #[inline]
    pub fn relative_path_ends_with(&self, suffix: &str) -> bool {
        let fname = self.file_name();
        if suffix.len() <= fname.len() {
            return fname.ends_with(suffix);
        }
        // suffix extends into dir portion
        let dir = self.dir_str();
        let total = dir.len() + fname.len();
        if suffix.len() > total {
            return false;
        }
        let dir_part = suffix.len() - fname.len();
        dir.ends_with(&suffix[..dir_part]) && fname == &suffix[dir_part..]
    }

    /// Check if the relative path starts with `prefix` without allocating.
    #[inline]
    pub fn relative_path_starts_with(&self, prefix: &str) -> bool {
        let dir = self.dir_str();
        if prefix.len() <= dir.len() {
            return dir.starts_with(prefix);
        }
        let fname = self.file_name();
        let total = dir.len() + fname.len();
        if prefix.len() > total {
            return false;
        }
        let fname_part = prefix.len() - dir.len();
        dir == &prefix[..dir.len()] && fname.starts_with(&prefix[dir.len()..dir.len() + fname_part])
    }

    /// Write the full relative path into a caller buffer. Zero-alloc.
    /// Returns the written `&str`, or panics if the buffer is too small.
    #[inline]
    pub fn write_relative_path<'a>(&self, buf: &'a mut [u8]) -> &'a str {
        let dir_len = self.dir_len as usize;
        let fname_len = self.filename_len as usize;
        let total = dir_len + fname_len;
        debug_assert!(buf.len() >= total, "buffer too small for relative path");
        unsafe {
            std::ptr::copy_nonoverlapping(self.dir_ptr, buf.as_mut_ptr(), dir_len);
            std::ptr::copy_nonoverlapping(
                self.filename_ptr,
                buf.as_mut_ptr().add(dir_len),
                fname_len,
            );
            std::str::from_utf8_unchecked(&buf[..total])
        }
    }

    /// Total byte length of the relative path (dir + filename).
    #[inline]
    pub fn relative_path_len(&self) -> usize {
        self.dir_len as usize + self.filename_len as usize
    }

    /// Just the filename component. Zero-cost slice into the filename arena.
    #[inline]
    pub fn file_name(&self) -> &str {
        unsafe {
            let slice = std::slice::from_raw_parts(self.filename_ptr, self.filename_len as usize);
            std::str::from_utf8_unchecked(slice)
        }
    }

    /// Raw directory bytes (exact length, no padding).
    #[inline]
    fn dir_bytes(&self) -> &[u8] {
        if self.dir_len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.dir_ptr, self.dir_len as usize) }
    }

    /// Raw filename bytes (exact length).
    #[inline]
    fn filename_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.filename_ptr, self.filename_len as usize) }
    }

    /// Byte offset of the filename within the relative path.
    /// This is the dir_len — same semantic as the old `filename_offset`.
    #[inline]
    pub fn filename_offset_in_relative(&self) -> usize {
        self.dir_len as usize
    }

    /// Reconstruct the full absolute path. Cold-path only (allocates).
    #[inline]
    pub fn absolute_path(&self, base_path: &Path) -> PathBuf {
        base_path.join(self.relative_path())
    }

    #[inline]
    pub fn total_frecency_score(&self) -> i32 {
        self.access_frecency_score as i32 + self.modification_frecency_score as i32
    }

    #[inline]
    pub fn is_binary(&self) -> bool {
        self.flags & FileItemFlags::BINARY != 0
    }

    #[inline]
    pub fn set_binary(&mut self, val: bool) {
        if val {
            self.flags |= FileItemFlags::BINARY;
        } else {
            self.flags &= !FileItemFlags::BINARY;
        }
    }

    #[inline]
    pub fn is_deleted(&self) -> bool {
        self.flags & FileItemFlags::DELETED != 0
    }

    #[inline]
    pub fn set_deleted(&mut self, val: bool) {
        if val {
            self.flags |= FileItemFlags::DELETED;
        } else {
            self.flags &= !FileItemFlags::DELETED;
        }
    }
}

// ── MatchableSegmented: zero-copy SIMD matching via [dir, filename] segments ──

impl MatchableSegmented for FileItem {
    #[inline]
    fn match_segments(&self) -> Option<([&[u8]; 2], u8)> {
        if self.is_deleted() {
            return None;
        }
        Some(([self.dir_bytes(), self.filename_bytes()], 2))
    }
}

impl MatchableSegmented for &FileItem {
    #[inline]
    fn match_segments(&self) -> Option<([&[u8]; 2], u8)> {
        (*self).match_segments()
    }
}

impl FileItem {
    /// Invalidate the cached content so the next `get_content()` call creates a fresh one.
    ///
    /// Call this when the background watcher detects that the file has been modified.
    /// On Unix, a file that is truncated while mapped can cause SIGBUS. On Windows,
    /// the stale buffer simply won't reflect the new contents. In both cases,
    /// invalidating ensures a fresh read on the next access.
    pub fn invalidate_mmap(&mut self, budget: &ContentCacheBudget) {
        if self.content.get().is_some() {
            budget.cached_count.fetch_sub(1, Ordering::Relaxed);
            budget.cached_bytes.fetch_sub(self.size, Ordering::Relaxed);
        }

        self.content = OnceLock::new();
    }

    /// Get the cached file contents or lazily load and cache them.
    ///
    /// Returns `None` if the file is too large, empty, can't be opened, **or
    /// the cache budget is exhausted**. Callers that need content regardless
    /// of the budget should use [`get_content_for_search`].
    ///
    /// After the first call, this is lock-free (just an atomic load + pointer deref).
    pub fn get_content(&self, base_path: &Path, budget: &ContentCacheBudget) -> Option<&[u8]> {
        if let Some(content) = self.content.get() {
            return Some(content);
        }

        let max_file_size = budget.max_file_size;
        if self.size == 0 || self.size > max_file_size {
            return None;
        }

        // Check cache budget before creating a new persistent cache entry.
        let count = budget.cached_count.load(Ordering::Relaxed);
        let bytes = budget.cached_bytes.load(Ordering::Relaxed);
        let max_files = budget.max_files;
        let max_bytes = budget.max_bytes;
        if count >= max_files || bytes + self.size > max_bytes {
            return None;
        }

        let content = load_file_content(&self.absolute_path(base_path), self.size)?;
        let result = self.content.get_or_init(|| content);

        // Bump counters. Slight over-count under races is fine — the budget
        // is a soft limit and the overshoot is bounded by rayon thread count.
        budget.cached_count.fetch_add(1, Ordering::Relaxed);
        budget.cached_bytes.fetch_add(self.size, Ordering::Relaxed);

        Some(result)
    }

    /// Get file content for searching — **always returns content** for eligible
    /// files, even when the persistent cache budget is exhausted.
    #[inline]
    pub fn get_content_for_search<'a>(
        &'a self,
        buf: &'a mut Vec<u8>,
        budget: &ContentCacheBudget,
    ) -> Option<&'a [u8]> {
        // Fast path: persistent cache hit (zero-copy).
        if let Some(cached) = self.get_content(budget) {
            return Some(cached);
        }

        let max_file_size = budget.max_file_size;
        if self.is_binary() || self.size == 0 || self.size > max_file_size {
            return None;
        }

        // Slow path: read into the reusable buffer — open() + read_exact() + close().
        // No mmap()/munmap() syscalls, no page table setup/teardown.
        // We know the exact size so we use read_exact (1 read syscall) instead of
        // read_to_end (2 read syscalls — one for data, one for EOF confirmation).
        let len = self.size as usize;
        buf.resize(len, 0);
        let mut file = std::fs::File::open(self.as_path()).ok()?;
        file.read_exact(buf).ok()?;
        Some(buf.as_slice())
    }
}

/// Page size on Apple Silicon is 16KB; on x86-64 it's 4KB.
/// Files smaller than one page waste the remainder when mmapped.
/// Reading them into a heap buffer avoids this overhead.
#[cfg(target_arch = "aarch64")]
const MMAP_THRESHOLD: u64 = 16 * 1024;
#[cfg(not(target_arch = "aarch64"))]
const MMAP_THRESHOLD: u64 = 4 * 1024;

/// Load file contents: small files are read into a heap buffer to avoid
/// mmap page alignment waste; large files use mmap for zero-copy access.
/// On Windows, always uses heap buffer (mmap holds the file handle open).
fn load_file_content(path: &Path, size: u64) -> Option<FileContent> {
    #[cfg(not(target_os = "windows"))]
    {
        if size < MMAP_THRESHOLD {
            let data = std::fs::read(path).ok()?;
            Some(FileContent::Buffer(data))
        } else {
            let file = std::fs::File::open(path).ok()?;
            // SAFETY: The mmap is backed by the kernel page cache and automatically
            // reflects file modifications. The only risk is SIGBUS if the file is
            // truncated while mapped.
            let mmap = unsafe { memmap2::Mmap::map(&file) }.ok()?;
            Some(FileContent::Mmap(mmap))
        }
    }

    #[cfg(target_os = "windows")]
    {
        let _ = size;
        let data = std::fs::read(path).ok()?;
        Some(FileContent::Buffer(data))
    }
}

impl Constrainable for FileItem {
    #[inline]
    fn dir_path(&self) -> &str {
        self.dir_str()
    }

    #[inline]
    fn file_name(&self) -> &str {
        FileItem::file_name(self)
    }

    #[inline]
    fn git_status(&self) -> Option<git2::Status> {
        self.git_status
    }
}

#[derive(Debug, Clone, Default)]
pub struct Score {
    pub total: i32,
    pub base_score: i32,
    pub filename_bonus: i32,
    pub special_filename_bonus: i32,
    pub frecency_boost: i32,
    pub git_status_boost: i32,
    pub distance_penalty: i32,
    pub current_file_penalty: i32,
    pub combo_match_boost: i32,
    pub path_alignment_bonus: i32,
    pub exact_match: bool,
    pub match_type: &'static str,
}

#[derive(Debug, Clone, Copy)]
pub struct PaginationArgs {
    pub offset: usize,
    pub limit: usize,
}

impl Default for PaginationArgs {
    fn default() -> Self {
        Self {
            offset: 0,
            limit: 100,
        }
    }
}

/// Context for scoring files during search.
///
/// The `query` field contains the pre-parsed query with constraints,
/// fuzzy parts, and location information. Parsing is done once at the API
/// boundary and passed through.
#[derive(Debug, Clone)]
pub struct ScoringContext<'a> {
    /// Parsed query containing raw text, constraints, fuzzy parts, and location
    pub query: &'a FFFQuery<'a>,
    pub project_path: Option<&'a Path>,
    pub current_file: Option<&'a str>,
    pub max_typos: u16,
    pub max_threads: usize,
    pub last_same_query_match: Option<QueryMatchEntry>,
    pub combo_boost_score_multiplier: i32,
    pub min_combo_count: u32,
    pub pagination: PaginationArgs,
    /// Path bigram index for pre-filtering fuzzy search candidates.
    /// When present, eliminates ~90% of files before the expensive SIMD matching.
    pub path_bigram_index: Option<&'a crate::bigram_filter::BigramFilter>,
}

impl ScoringContext<'_> {
    /// Get the effective fuzzy query string for matching.
    /// Returns the first fuzzy part, or the raw query if no parsing was done.
    pub fn effective_query(&self) -> &str {
        match &self.query.fuzzy_query {
            FuzzyQuery::Text(t) => t,
            FuzzyQuery::Parts(parts) if !parts.is_empty() => parts[0],
            _ => self.query.raw_query.trim(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SearchResult<'a> {
    pub items: Vec<&'a FileItem>,
    pub scores: Vec<Score>,
    pub total_matched: usize,
    pub total_files: usize,
    pub location: Option<Location>,
}

const MAX_MMAP_FILE_SIZE: u64 = 10 * 1024 * 1024;

// Limits the total number of files (and bytes) whose content is kept in
// memory via the `OnceLock<FileContent>` cache. On Unix every cached file
// holds a live `mmap`, which consumes a kernel `vm_map_entry`. On a 500k-file
// monorepo, caching everything exhausts macOS/Linux kernel resources and
// crashes the machine (see issue #294).
//
// Each `FilePicker` owns its own `ContentCacheBudget`. The budget is passed
// to `grep_search` and `warmup_mmaps` so that multiple pickers can coexist
// without interfering with each other's counters.

const MAX_CACHED_CONTENT_BYTES: u64 = 512 * 1024 * 1024;

/// Per-picker budget controlling how many files may have their content
/// persistently cached (mmap on Unix, heap buffer on Windows).
#[derive(Debug)]
pub struct ContentCacheBudget {
    pub max_files: usize,
    pub max_bytes: u64,
    pub max_file_size: u64,
    pub cached_count: AtomicUsize,
    pub cached_bytes: AtomicU64,
}

impl ContentCacheBudget {
    /// No limits — every eligible file is cached. Useful for tests and
    /// short-lived tools that don't need resource protection.
    pub fn unlimited() -> Self {
        Self {
            max_files: usize::MAX,
            max_bytes: u64::MAX,
            max_file_size: MAX_MMAP_FILE_SIZE,
            cached_count: AtomicUsize::new(0),
            cached_bytes: AtomicU64::new(0),
        }
    }

    pub fn zero() -> Self {
        Self {
            max_files: 0,
            max_bytes: 0,
            max_file_size: 0,
            cached_count: AtomicUsize::new(0),
            cached_bytes: AtomicU64::new(0),
        }
    }

    pub fn new_for_repo(file_count: usize) -> Self {
        let max_files = if file_count > 50_000 {
            5_000
        } else if file_count > 10_000 {
            10_000
        } else {
            30_000 // effectively unlimited for small repos
        };

        let max_bytes = if file_count > 50_000 {
            128 * 1024 * 1024 // 128 MB
        } else if file_count > 10_000 {
            256 * 1024 * 1024 // 256 MB
        } else {
            MAX_CACHED_CONTENT_BYTES // 512 MB
        };

        Self {
            max_files,
            max_bytes,
            max_file_size: MAX_MMAP_FILE_SIZE,
            cached_count: AtomicUsize::new(0),
            cached_bytes: AtomicU64::new(0),
        }
    }

    /// Reset the counters. Called when the file index is rebuilt (rescan /
    /// directory change) and all old `FileItem`s are dropped.
    pub fn reset(&self) {
        self.cached_count.store(0, Ordering::Relaxed);
        self.cached_bytes.store(0, Ordering::Relaxed);
    }
}

impl Default for ContentCacheBudget {
    fn default() -> Self {
        Self::new_for_repo(30_000)
    }
}
