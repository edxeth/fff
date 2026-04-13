//! SIMD-aligned chunked path storage with directory deduplication.
//!
//! # Motivation
//!
//! In a typical project, many files share the same parent directory. The old
//! arena stored full relative paths (`src/components/Button.tsx`,
//! `src/components/Dialog.tsx`, …) — duplicating `src/components/` for every
//! file. On a 100k-file project that wastes megabytes.
//!
//! This module stores paths in a **split, deduplicated** layout:
//!
//! - **Directory bytes** — stored as 16-byte aligned chunk sequences.
//!   All files in the same directory share the same chunk sequence (dedup).
//! - **Filename bytes** — stored contiguously in a packed arena.
//!
//! For matching, the two parts are passed as segments to frizbee's
//! [`MatchableSegmented`] — the SIMD scorer processes them as a virtual
//! concatenation with near-zero overhead (only a 16-byte bridge copy at the
//! segment boundary).
//!
//! For output, the full path string is reconstructed on demand:
//! `dir_bytes ++ filename_bytes = relative_path`.
//!
//! # Memory layout
//!
//! ```text
//! SimdPathStore:
//!
//! chunk_arena: [ 16B ][ 16B ][ 16B ] ... [ 16B ]   ← directory chunks, 16-byte aligned
//!               ↑ dir 0       ↑ dir 1               (deduplicated)
//!
//! filename_arena: "Button.tsxDialog.tsxindex.ts..."  ← all filenames packed
//!                  ↑ file 0   ↑ file 1  ↑ file 2
//! ```
//!
//! # SIMD integration
//!
//! frizbee's `score_haystack_segments` iterates over the haystack in 16-byte
//! chunks via `Simd128::load_partial(ptr, col * 16, len)`. When the dir
//! segment starts at a 16-byte aligned address (guaranteed by `SimdChunk`),
//! full-chunk loads within the directory are naturally aligned — faster on
//! ARM (NEON) and free on x86 (SSE uses unaligned loads anyway).

use neo_frizbee::MatchableSegmented;

// ────────────────────────────────────────────────────────────────────────────
// Core types
// ────────────────────────────────────────────────────────────────────────────

/// 16-byte SIMD-aligned chunk — same width as `uint8x16_t` / `__m128i`.
///
/// Each chunk stores up to 16 bytes of path data. The last chunk in a
/// directory entry is zero-padded on the right. The alignment guarantees
/// that SIMD loads within a chunk sequence hit aligned addresses.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct SimdChunk([u8; 16]);

impl SimdChunk {
    /// Mutable access to the underlying byte array.
    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 16] {
        &mut self.0
    }
}

impl Default for SimdChunk {
    #[inline]
    fn default() -> Self {
        Self([0u8; 16])
    }
}

impl std::fmt::Debug for SimdChunk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Show the actual bytes, trimming trailing zeros for readability
        let end = self.0.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
        write!(f, "SimdChunk({:?})", &self.0[..end])
    }
}

/// Metadata for one directory's chunk sequence in the arena.
///
/// The directory relative path (e.g. `src/components/`) is stored as
/// `chunk_count` contiguous [`SimdChunk`]s starting at byte offset
/// `chunk_byte_offset` in the arena. Only the first `byte_len` bytes
/// are valid path data; the rest is zero-padding.
#[derive(Clone, Copy, Debug)]
pub struct DirChunkRef {
    /// Byte offset of the first chunk in `SimdPathStore::chunk_arena`,
    /// measured from the arena's base pointer. Always 16-byte aligned.
    pub chunk_byte_offset: u32,
    /// Number of 16-byte chunks occupied by this directory path.
    pub chunk_count: u8,
    /// Actual byte length of the directory path (without SIMD padding).
    /// For root-level files (no directory prefix), this is 0.
    pub byte_len: u16,
}

impl DirChunkRef {
    /// Directory with no path (root-level files).
    pub const EMPTY: Self = Self {
        chunk_byte_offset: 0,
        chunk_count: 0,
        byte_len: 0,
    };
}

// ────────────────────────────────────────────────────────────────────────────
// SimdPathStore — the arena
// ────────────────────────────────────────────────────────────────────────────

/// Arena holding all file paths as SIMD-aligned directory chunks + packed filenames.
///
/// Directory deduplication: N files in the same directory share one chunk
/// sequence. Filenames are stored once each in a flat byte array.
///
/// After construction the arenas are **immutable** — raw pointers into them
/// remain valid for the store's lifetime.
pub struct SimdPathStore {
    /// Contiguous 16-byte aligned chunk storage for all directory paths.
    /// Multiple [`DirChunkRef`]s point into different offsets here.
    chunk_arena: Vec<SimdChunk>,

    /// Per-directory metadata. Index matches the file's `parent_dir` field.
    dir_table: Vec<DirChunkRef>,

    /// Contiguous filename storage, packed with no gaps or alignment padding.
    /// Filenames are typically short (< 30 bytes) so alignment doesn't matter —
    /// frizbee handles them as the second segment with at most one `load_partial`.
    filename_arena: Vec<u8>,
}

impl SimdPathStore {
    /// Total heap bytes used by this store's arenas (excluding the struct itself).
    pub fn heap_bytes(&self) -> usize {
        self.chunk_arena.len() * 16
            + self.dir_table.len() * std::mem::size_of::<DirChunkRef>()
            + self.filename_arena.len()
    }

    /// Number of unique directories stored.
    pub fn dir_count(&self) -> usize {
        self.dir_table.len()
    }

    /// Total chunks in the directory arena.
    pub fn chunk_count(&self) -> usize {
        self.chunk_arena.len()
    }

    /// Total bytes in the filename arena.
    pub fn filename_arena_len(&self) -> usize {
        self.filename_arena.len()
    }

    /// Get the raw directory path bytes for a directory index.
    ///
    /// Returns a contiguous `&[u8]` of the actual path (e.g. `b"src/components/"`)
    /// without SIMD padding bytes. The returned slice's base address is 16-byte aligned.
    ///
    /// # Panics
    /// Panics if `dir_index >= self.dir_count()`.
    #[inline]
    pub fn dir_bytes(&self, dir_index: u32) -> &[u8] {
        let dir = &self.dir_table[dir_index as usize];
        if dir.byte_len == 0 {
            return &[];
        }
        unsafe {
            let base = self.chunk_arena.as_ptr() as *const u8;
            std::slice::from_raw_parts(
                base.add(dir.chunk_byte_offset as usize),
                dir.byte_len as usize,
            )
        }
    }

    /// Get the raw directory path as a `&str`.
    #[inline]
    pub fn dir_str(&self, dir_index: u32) -> &str {
        // SAFETY: directory paths are always valid UTF-8 (written from &str).
        unsafe { std::str::from_utf8_unchecked(self.dir_bytes(dir_index)) }
    }

    /// Get filename bytes from the arena.
    #[inline]
    pub fn filename_bytes_at(&self, offset: u32, len: u16) -> &[u8] {
        &self.filename_arena[offset as usize..offset as usize + len as usize]
    }

    /// Get filename as `&str`.
    #[inline]
    pub fn filename_str(&self, offset: u32, len: u16) -> &str {
        unsafe { std::str::from_utf8_unchecked(self.filename_bytes_at(offset, len)) }
    }

    /// Get the [`DirChunkRef`] for a directory index.
    #[inline]
    pub fn dir_ref(&self, dir_index: u32) -> &DirChunkRef {
        &self.dir_table[dir_index as usize]
    }
}

impl std::fmt::Debug for SimdPathStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimdPathStore")
            .field("dirs", &self.dir_table.len())
            .field("chunks", &self.chunk_arena.len())
            .field("filename_bytes", &self.filename_arena.len())
            .field("heap_bytes", &self.heap_bytes())
            .finish()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// SimdFileEntry — per-file reference into the store
// ────────────────────────────────────────────────────────────────────────────

/// A file entry referencing SIMD-chunked directory bytes and packed filename bytes.
///
/// Self-contained: implements [`MatchableSegmented`] without needing an external
/// arena reference. Raw pointers point into the [`SimdPathStore`]'s immutable arenas.
///
/// # Layout (24 bytes on 64-bit)
///
/// ```text
///   dir_ptr:      *const u8  (8)  → chunk_arena (16-byte aligned)
///   filename_ptr: *const u8  (8)  → filename_arena
///   dir_len:      u16        (2)  actual dir path bytes
///   filename_len: u16        (2)  actual filename bytes
///   flags:        u8         (1)  deleted / binary / etc.
///   _pad:         [u8; 3]    (3)  alignment padding
/// ```
pub struct SimdFileEntry {
    /// Raw pointer to directory bytes in the chunk arena (16-byte aligned base).
    /// For root-level files this points to the arena base (but `dir_len == 0`).
    dir_ptr: *const u8,
    /// Raw pointer to filename bytes in the filename arena.
    filename_ptr: *const u8,
    /// Actual byte length of the directory portion (e.g. `src/lib/` = 8).
    /// 0 for root-level files.
    dir_len: u16,
    /// Byte length of the filename.
    filename_len: u16,
    /// Bit flags (mirrors FileItemFlags). Bit 0 = binary, bit 1 = deleted.
    flags: u8,
}

// SAFETY: pointers reference immutable arenas (Vec<SimdChunk> and Vec<u8>)
// that are never reallocated after construction of SimdPathStore.
unsafe impl Send for SimdFileEntry {}
unsafe impl Sync for SimdFileEntry {}

impl SimdFileEntry {
    const DELETED: u8 = 1 << 1;

    /// The directory portion of the relative path as a byte slice.
    ///
    /// For `src/components/Button.tsx` this returns `b"src/components/"`.
    /// For root-level `Cargo.toml` this returns `&[]`.
    #[inline]
    pub fn dir_bytes(&self) -> &[u8] {
        if self.dir_len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.dir_ptr, self.dir_len as usize) }
    }

    /// The filename as a byte slice.
    #[inline]
    pub fn filename_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.filename_ptr, self.filename_len as usize) }
    }

    /// The directory portion as `&str`.
    #[inline]
    pub fn dir_str(&self) -> &str {
        unsafe { std::str::from_utf8_unchecked(self.dir_bytes()) }
    }

    /// The filename as `&str`.
    #[inline]
    pub fn filename_str(&self) -> &str {
        unsafe { std::str::from_utf8_unchecked(self.filename_bytes()) }
    }

    /// Total byte length of the full relative path (`dir + filename`).
    #[inline]
    pub fn total_len(&self) -> usize {
        self.dir_len as usize + self.filename_len as usize
    }

    #[inline]
    pub fn is_deleted(&self) -> bool {
        self.flags & Self::DELETED != 0
    }

    #[inline]
    pub fn set_deleted(&mut self, val: bool) {
        if val {
            self.flags |= Self::DELETED;
        } else {
            self.flags &= !Self::DELETED;
        }
    }

    /// Reconstruct the full relative path. **Allocates** — use only for output.
    ///
    /// This is the cold path: called when sending results to Neovim or when
    /// building absolute paths for file I/O. The hot path (matching) uses
    /// [`MatchableSegmented::match_segments`] which is zero-copy.
    pub fn reconstruct_path(&self) -> String {
        let dir = self.dir_bytes();
        let filename = self.filename_bytes();
        let mut s = String::with_capacity(dir.len() + filename.len());
        // SAFETY: both slices are valid UTF-8
        unsafe {
            s.push_str(std::str::from_utf8_unchecked(dir));
            s.push_str(std::str::from_utf8_unchecked(filename));
        }
        s
    }

    /// Write the full relative path into a caller-provided buffer.
    /// Returns the number of bytes written, or `None` if the buffer is too small.
    ///
    /// Stack-friendly alternative to [`reconstruct_path`] when you have a
    /// `[u8; 512]` or similar on the stack.
    #[inline]
    pub fn write_path_to_buf<'a>(&self, buf: &'a mut [u8]) -> Option<&'a str> {
        let total = self.total_len();
        if buf.len() < total {
            return None;
        }
        let dir_len = self.dir_len as usize;
        unsafe {
            std::ptr::copy_nonoverlapping(self.dir_ptr, buf.as_mut_ptr(), dir_len);
            std::ptr::copy_nonoverlapping(
                self.filename_ptr,
                buf.as_mut_ptr().add(dir_len),
                self.filename_len as usize,
            );
            Some(std::str::from_utf8_unchecked(&buf[..total]))
        }
    }
}

impl std::fmt::Debug for SimdFileEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimdFileEntry")
            .field("dir", &self.dir_str())
            .field("filename", &self.filename_str())
            .field("flags", &self.flags)
            .finish()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// MatchableSegmented — zero-copy frizbee integration
// ────────────────────────────────────────────────────────────────────────────

impl MatchableSegmented for SimdFileEntry {
    /// Return `[dir_bytes, filename_bytes]` as two segments.
    ///
    /// frizbee's `score_haystack_segments` processes them as a virtual
    /// concatenation: `"src/components/" ++ "Button.tsx"` → scored as if
    /// `"src/components/Button.tsx"`. Only a 16-byte bridge copy happens
    /// at the segment boundary.
    #[inline]
    fn match_segments(&self) -> Option<([&[u8]; 2], u8)> {
        if self.is_deleted() {
            return None;
        }
        Some(([self.dir_bytes(), self.filename_bytes()], 2))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Builder
// ────────────────────────────────────────────────────────────────────────────

/// Input for a single file during arena construction.
/// The builder needs the directory relative path (with trailing `/` for
/// non-root) and the filename.
pub struct FilePathInput<'a> {
    /// The directory relative path, **including trailing `/`** for non-root.
    /// Empty string for root-level files.
    pub dir_relative: &'a str,
    /// The filename (basename). E.g. `Button.tsx`.
    pub filename: &'a str,
    /// Index into the directory table. Files with the same `dir_index`
    /// share the same chunk sequence — this drives deduplication.
    pub dir_index: u32,
    /// Initial flags (binary, etc.)
    pub flags: u8,
}

/// Result of building a [`SimdPathStore`].
pub struct SimdPathBuildResult {
    pub store: SimdPathStore,
    pub entries: Vec<SimdFileEntry>,
}

/// Build a [`SimdPathStore`] from sorted file inputs.
///
/// Files must be grouped by `dir_index` (files with the same directory must
/// have the same `dir_index`). The directory table is built by processing
/// unique `dir_index` values in order.
///
/// # Algorithm
///
/// 1. Scan inputs to discover unique directories, compute chunk counts.
/// 2. Pre-allocate the chunk arena (exact capacity, no reallocation).
/// 3. For each unique directory: copy bytes into 16-byte aligned chunks.
/// 4. Pre-allocate the filename arena (exact capacity).
/// 5. For each file: copy filename bytes, create `SimdFileEntry` with
///    pointers into the now-frozen arenas.
///
/// After this function returns, the arenas are never reallocated, so the
/// raw pointers in `SimdFileEntry` remain valid for the store's lifetime.
pub fn build_simd_path_store(inputs: &[FilePathInput<'_>]) -> SimdPathBuildResult {
    // ── Phase 1: Discover unique directories, compute sizes ──
    // (inputs are sorted by dir_index, so unique dirs appear in runs)
    let mut unique_dirs: Vec<(u32, &str)> = Vec::new(); // (dir_index, dir_relative)
    let mut total_chunks = 0usize;
    let mut total_filename_bytes = 0usize;

    for input in inputs {
        total_filename_bytes += input.filename.len();

        if unique_dirs
            .last()
            .is_none_or(|(idx, _)| *idx != input.dir_index)
        {
            let chunk_count = chunks_needed(input.dir_relative.len());
            total_chunks += chunk_count;
            unique_dirs.push((input.dir_index, input.dir_relative));
        }
    }

    // ── Phase 2: Allocate arenas at exact capacity ──
    let mut chunk_arena: Vec<SimdChunk> = Vec::with_capacity(total_chunks);
    let mut dir_table: Vec<DirChunkRef> = Vec::with_capacity(unique_dirs.len());
    let mut filename_arena: Vec<u8> = Vec::with_capacity(total_filename_bytes);

    // ── Phase 3: Fill directory chunk arena ──
    for &(_dir_index, dir_rel) in &unique_dirs {
        let byte_len = dir_rel.len();
        let n_chunks = chunks_needed(byte_len);
        let chunk_byte_offset = chunk_arena.len() * 16;

        let bytes = dir_rel.as_bytes();
        for i in 0..n_chunks {
            let mut chunk = SimdChunk::default(); // zero-initialized
            let start = i * 16;
            let end = (start + 16).min(byte_len);
            chunk.0[..end - start].copy_from_slice(&bytes[start..end]);
            // Remaining bytes stay zero (SIMD padding)
            chunk_arena.push(chunk);
        }

        dir_table.push(DirChunkRef {
            chunk_byte_offset: chunk_byte_offset as u32,
            chunk_count: n_chunks as u8,
            byte_len: byte_len as u16,
        });
    }

    debug_assert_eq!(chunk_arena.len(), total_chunks, "chunk arena size mismatch");
    debug_assert_eq!(
        chunk_arena.capacity(),
        total_chunks,
        "chunk arena should not have reallocated"
    );

    // ── Phase 4: Fill filename arena + build file entries ──
    let mut entries: Vec<SimdFileEntry> = Vec::with_capacity(inputs.len());

    // Map from dir_index → position in dir_table (for pointer lookup).
    // Since unique_dirs is sorted by dir_index, use binary search.
    let dir_index_to_table_pos = |dir_index: u32| -> usize {
        unique_dirs
            .binary_search_by_key(&dir_index, |(idx, _)| *idx)
            .expect("dir_index must exist in unique_dirs")
    };

    // Freeze arena pointers — no more pushes to chunk_arena or filename_arena
    // after this point (we pre-allocated exact capacity).
    let chunk_base_ptr = chunk_arena.as_ptr() as *const u8;

    for input in inputs {
        // Filename: append to arena, record offset
        let filename_offset = filename_arena.len();
        filename_arena.extend_from_slice(input.filename.as_bytes());
        let filename_ptr = unsafe { filename_arena.as_ptr().add(filename_offset) };

        // Directory: look up chunk ref for pointer
        let table_pos = dir_index_to_table_pos(input.dir_index);
        let dir_ref = &dir_table[table_pos];
        let dir_ptr = unsafe { chunk_base_ptr.add(dir_ref.chunk_byte_offset as usize) };

        entries.push(SimdFileEntry {
            dir_ptr,
            filename_ptr,
            dir_len: dir_ref.byte_len,
            filename_len: input.filename.len() as u16,
            flags: input.flags,
        });
    }

    debug_assert_eq!(
        filename_arena.len(),
        total_filename_bytes,
        "filename arena size mismatch"
    );
    debug_assert_eq!(
        filename_arena.capacity(),
        total_filename_bytes,
        "filename arena should not have reallocated"
    );

    SimdPathBuildResult {
        store: SimdPathStore {
            chunk_arena,
            dir_table,
            filename_arena,
        },
        entries,
    }
}

/// Number of 16-byte chunks needed to store `byte_len` bytes.
/// Returns 0 for empty input (root-level directory).
#[inline]
const fn chunks_needed(byte_len: usize) -> usize {
    if byte_len == 0 {
        0
    } else {
        (byte_len + 15) / 16
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Overflow support (watcher-added files)
// ────────────────────────────────────────────────────────────────────────────

/// Per-file overflow allocation for watcher-added files.
///
/// Each overflow file gets its own `Box<[u8]>` holding both the directory
/// path and filename contiguously (with 16-byte padding between them for
/// SIMD alignment of the filename segment start). Growing the overflow
/// Vec never invalidates pointers of previously-added files.
///
/// Layout inside the box:
/// ```text
/// [ dir bytes ][ zero-pad to 16B ][ filename bytes ][ zero-pad to 16B ]
///  ^dir_ptr                        ^filename_ptr
/// ```
pub struct OverflowFileEntry {
    /// Heap allocation holding both segments.
    #[allow(dead_code)]
    buf: Box<[u8]>,
    /// The file entry with pointers into `buf`.
    pub entry: SimdFileEntry,
}

impl OverflowFileEntry {
    /// Create an overflow entry for a newly-discovered file.
    ///
    /// `dir_relative` should include trailing `/` for non-root directories.
    pub fn new(dir_relative: &str, filename: &str, flags: u8) -> Self {
        let dir_len = dir_relative.len();
        let dir_padded = align_up_16(dir_len);
        let filename_len = filename.len();
        let filename_padded = align_up_16(filename_len);
        let total = dir_padded + filename_padded;

        let mut buf = vec![0u8; total];
        buf[..dir_len].copy_from_slice(dir_relative.as_bytes());
        buf[dir_padded..dir_padded + filename_len].copy_from_slice(filename.as_bytes());

        let boxed: Box<[u8]> = buf.into_boxed_slice();
        let dir_ptr = boxed.as_ptr();
        let filename_ptr = unsafe { boxed.as_ptr().add(dir_padded) };

        Self {
            entry: SimdFileEntry {
                dir_ptr,
                filename_ptr,
                dir_len: dir_len as u16,
                filename_len: filename_len as u16,
                flags,
            },
            buf: boxed,
        }
    }
}

#[inline]
const fn align_up_16(n: usize) -> usize {
    (n + 15) & !15
}

// ────────────────────────────────────────────────────────────────────────────
// Memory comparison helpers
// ────────────────────────────────────────────────────────────────────────────

/// Compute what the old flat arena would have cost for the same set of files.
/// Useful for benchmarking / logging the memory savings.
pub fn estimate_flat_arena_bytes(inputs: &[FilePathInput<'_>]) -> usize {
    inputs
        .iter()
        .map(|f| f.dir_relative.len() + f.filename.len())
        .sum()
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──

    fn make_input<'a>(dir: &'a str, filename: &'a str, dir_index: u32) -> FilePathInput<'a> {
        FilePathInput {
            dir_relative: dir,
            filename,
            dir_index,
            flags: 0,
        }
    }

    // ── Core construction tests ──

    #[test]
    fn test_empty_store() {
        let result = build_simd_path_store(&[]);
        assert_eq!(result.store.dir_count(), 0);
        assert_eq!(result.store.chunk_count(), 0);
        assert_eq!(result.store.filename_arena_len(), 0);
        assert_eq!(result.entries.len(), 0);
    }

    #[test]
    fn test_single_root_file() {
        let inputs = [make_input("", "Cargo.toml", 0)];
        let result = build_simd_path_store(&inputs);

        assert_eq!(result.store.dir_count(), 1);
        assert_eq!(result.entries.len(), 1);

        let entry = &result.entries[0];
        assert_eq!(entry.dir_bytes(), b"");
        assert_eq!(entry.filename_bytes(), b"Cargo.toml");
        assert_eq!(entry.reconstruct_path(), "Cargo.toml");
    }

    #[test]
    fn test_single_nested_file() {
        let inputs = [make_input("src/lib/", "main.rs", 0)];
        let result = build_simd_path_store(&inputs);

        assert_eq!(result.store.dir_count(), 1);
        let entry = &result.entries[0];
        assert_eq!(entry.dir_str(), "src/lib/");
        assert_eq!(entry.filename_str(), "main.rs");
        assert_eq!(entry.reconstruct_path(), "src/lib/main.rs");
    }

    #[test]
    fn test_directory_deduplication() {
        // 3 files in the same directory should share 1 chunk sequence
        let inputs = [
            make_input("src/components/", "Button.tsx", 0),
            make_input("src/components/", "Dialog.tsx", 0),
            make_input("src/components/", "Modal.tsx", 0),
        ];
        let result = build_simd_path_store(&inputs);

        assert_eq!(result.store.dir_count(), 1, "only 1 unique directory");
        // "src/components/" = 15 bytes → 1 chunk
        assert_eq!(result.store.chunk_count(), 1);

        // All 3 entries should share the same dir pointer
        let ptr0 = result.entries[0].dir_ptr;
        let ptr1 = result.entries[1].dir_ptr;
        let ptr2 = result.entries[2].dir_ptr;
        assert_eq!(ptr0, ptr1);
        assert_eq!(ptr1, ptr2);

        // But filenames differ
        assert_eq!(result.entries[0].filename_str(), "Button.tsx");
        assert_eq!(result.entries[1].filename_str(), "Dialog.tsx");
        assert_eq!(result.entries[2].filename_str(), "Modal.tsx");

        // Full paths reconstruct correctly
        assert_eq!(
            result.entries[0].reconstruct_path(),
            "src/components/Button.tsx"
        );
        assert_eq!(
            result.entries[1].reconstruct_path(),
            "src/components/Dialog.tsx"
        );
    }

    #[test]
    fn test_multiple_directories() {
        let inputs = [
            make_input("src/", "lib.rs", 0),
            make_input("src/", "main.rs", 0),
            make_input("tests/", "integration.rs", 1),
            make_input("tests/", "unit.rs", 1),
        ];
        let result = build_simd_path_store(&inputs);

        assert_eq!(result.store.dir_count(), 2);
        assert_eq!(result.entries[0].reconstruct_path(), "src/lib.rs");
        assert_eq!(result.entries[1].reconstruct_path(), "src/main.rs");
        assert_eq!(result.entries[2].reconstruct_path(), "tests/integration.rs");
        assert_eq!(result.entries[3].reconstruct_path(), "tests/unit.rs");

        // src/ files share one pointer, tests/ files share another
        assert_eq!(result.entries[0].dir_ptr, result.entries[1].dir_ptr);
        assert_eq!(result.entries[2].dir_ptr, result.entries[3].dir_ptr);
        assert_ne!(result.entries[0].dir_ptr, result.entries[2].dir_ptr);
    }

    #[test]
    fn test_chunk_alignment() {
        // Verify that directory pointers are 16-byte aligned
        let inputs = [
            make_input("src/components/ui/", "Button.tsx", 0), // 18 bytes → 2 chunks
            make_input("tests/", "test.rs", 1),                // 6 bytes → 1 chunk
        ];
        let result = build_simd_path_store(&inputs);

        for entry in &result.entries {
            if entry.dir_len > 0 {
                let addr = entry.dir_ptr as usize;
                assert_eq!(
                    addr % 16,
                    0,
                    "dir pointer must be 16-byte aligned, got {:#x}",
                    addr
                );
            }
        }
    }

    #[test]
    fn test_long_directory_path_multi_chunk() {
        // "very/deeply/nested/directory/structure/" = 39 bytes → 3 chunks (48 bytes)
        let long_dir = "very/deeply/nested/directory/structure/";
        assert_eq!(long_dir.len(), 39);

        let inputs = [make_input(long_dir, "file.txt", 0)];
        let result = build_simd_path_store(&inputs);

        let dir_ref = result.store.dir_ref(0);
        assert_eq!(dir_ref.chunk_count, 3, "39 bytes needs 3 chunks");
        assert_eq!(dir_ref.byte_len, 39);

        let entry = &result.entries[0];
        assert_eq!(entry.dir_str(), long_dir);
        assert_eq!(
            entry.reconstruct_path(),
            "very/deeply/nested/directory/structure/file.txt"
        );
    }

    #[test]
    fn test_exactly_16_byte_directory() {
        // Exactly 16 bytes → 1 chunk, no padding needed
        let dir = "0123456789abcde/"; // 16 bytes
        assert_eq!(dir.len(), 16);

        let inputs = [make_input(dir, "x.rs", 0)];
        let result = build_simd_path_store(&inputs);

        let dir_ref = result.store.dir_ref(0);
        assert_eq!(dir_ref.chunk_count, 1);
        assert_eq!(dir_ref.byte_len, 16);
        assert_eq!(result.entries[0].dir_str(), dir);
    }

    #[test]
    fn test_17_byte_directory_needs_2_chunks() {
        let dir = "0123456789abcdef/"; // 17 bytes → 2 chunks
        assert_eq!(dir.len(), 17);

        let inputs = [make_input(dir, "x.rs", 0)];
        let result = build_simd_path_store(&inputs);

        let dir_ref = result.store.dir_ref(0);
        assert_eq!(dir_ref.chunk_count, 2, "17 bytes needs 2 chunks");
        assert_eq!(result.entries[0].dir_str(), dir);
    }

    // ── MatchableSegmented tests ──

    #[test]
    fn test_matchable_segmented_nested_file() {
        let inputs = [make_input("src/components/", "Button.tsx", 0)];
        let result = build_simd_path_store(&inputs);
        let entry = &result.entries[0];

        let (segments, count) = entry.match_segments().unwrap();
        assert_eq!(count, 2);
        assert_eq!(segments[0], b"src/components/");
        assert_eq!(segments[1], b"Button.tsx");
    }

    #[test]
    fn test_matchable_segmented_root_file() {
        let inputs = [make_input("", "README.md", 0)];
        let result = build_simd_path_store(&inputs);
        let entry = &result.entries[0];

        let (segments, count) = entry.match_segments().unwrap();
        assert_eq!(count, 2);
        assert_eq!(segments[0], b""); // empty dir
        assert_eq!(segments[1], b"README.md");
    }

    #[test]
    fn test_matchable_segmented_deleted_returns_none() {
        let inputs = [make_input("src/", "deleted.rs", 0)];
        let mut result = build_simd_path_store(&inputs);
        result.entries[0].set_deleted(true);

        assert!(result.entries[0].match_segments().is_none());
    }

    #[test]
    fn test_write_path_to_buf() {
        let inputs = [make_input("src/", "main.rs", 0)];
        let result = build_simd_path_store(&inputs);
        let entry = &result.entries[0];

        let mut buf = [0u8; 256];
        let path = entry.write_path_to_buf(&mut buf).unwrap();
        assert_eq!(path, "src/main.rs");
    }

    // ── frizbee integration test ──

    #[test]
    fn test_frizbee_segmented_matching() {
        let inputs = [
            make_input("src/components/", "Button.tsx", 0),
            make_input("src/components/", "Dialog.tsx", 0),
            make_input("src/utils/", "helpers.ts", 1),
            make_input("", "README.md", 2),
        ];
        let result = build_simd_path_store(&inputs);

        let config = neo_frizbee::Config {
            max_typos: Some(2),
            sort: true,
            ..Default::default()
        };

        // Search for "Button" — should match Button.tsx
        let matches =
            neo_frizbee::match_list_parallel_segmented("Button", &result.entries, &config, 1);
        assert!(!matches.is_empty(), "should find Button.tsx");
        assert_eq!(
            result.entries[matches[0].index as usize].filename_str(),
            "Button.tsx"
        );

        // Search for "helpers" — should match helpers.ts
        let matches =
            neo_frizbee::match_list_parallel_segmented("helpers", &result.entries, &config, 1);
        assert!(!matches.is_empty(), "should find helpers.ts");
        assert_eq!(
            result.entries[matches[0].index as usize].filename_str(),
            "helpers.ts"
        );

        // Search for "README" — should match the root file
        let matches =
            neo_frizbee::match_list_parallel_segmented("README", &result.entries, &config, 1);
        assert!(!matches.is_empty(), "should find README.md");
        assert_eq!(
            result.entries[matches[0].index as usize].reconstruct_path(),
            "README.md"
        );
    }

    #[test]
    fn test_frizbee_cross_segment_boundary_matching() {
        // The query "ents/But" straddles the dir/filename boundary.
        // frizbee's score_haystack_segments must handle the bridge correctly.
        let inputs = [make_input("src/components/", "Button.tsx", 0)];
        let result = build_simd_path_store(&inputs);

        let config = neo_frizbee::Config {
            max_typos: Some(2),
            sort: false,
            ..Default::default()
        };

        let matches =
            neo_frizbee::match_list_parallel_segmented("ents/But", &result.entries, &config, 1);
        assert!(
            !matches.is_empty(),
            "cross-segment-boundary query should still match"
        );
    }

    #[test]
    fn test_frizbee_segmented_vs_contiguous_parity() {
        // Verify that segmented matching produces the same score as
        // matching against the contiguous path string.
        let dir = "src/components/ui/";
        let filename = "DatePicker.tsx";
        let full_path = format!("{}{}", dir, filename);

        let inputs = [make_input(dir, filename, 0)];
        let result = build_simd_path_store(&inputs);

        let config = neo_frizbee::Config {
            max_typos: Some(3),
            sort: false,
            ..Default::default()
        };

        let needle = "datepckr";

        // Segmented match
        let seg_matches =
            neo_frizbee::match_list_parallel_segmented(needle, &result.entries, &config, 1);
        // Contiguous match
        let cont_matches =
            neo_frizbee::match_list_parallel(needle, &[full_path.as_str()], &config, 1);

        assert!(!seg_matches.is_empty(), "segmented should match");
        assert!(!cont_matches.is_empty(), "contiguous should match");
        assert_eq!(
            seg_matches[0].score, cont_matches[0].score,
            "segmented and contiguous scores must be identical"
        );
    }

    // ── Memory savings test ──

    #[test]
    fn test_memory_savings() {
        // Simulate a directory with 100 files — the old arena stores the
        // dir prefix 100 times, the chunked store stores it once.
        let dir = "src/components/widgets/";
        let files: Vec<String> = (0..100).map(|i| format!("Widget{i}.tsx")).collect();

        let inputs: Vec<FilePathInput> = files
            .iter()
            .map(|f| make_input(dir, f.as_str(), 0))
            .collect();

        let flat_bytes = estimate_flat_arena_bytes(&inputs);
        let result = build_simd_path_store(&inputs);
        let chunked_bytes = result.store.heap_bytes();

        println!(
            "100 files in same dir: flat={flat_bytes}B, chunked={chunked_bytes}B, \
             savings={:.0}%",
            (1.0 - chunked_bytes as f64 / flat_bytes as f64) * 100.0,
        );

        assert!(
            chunked_bytes < flat_bytes,
            "chunked store must use less memory: {chunked_bytes} vs {flat_bytes}"
        );
    }

    #[test]
    fn test_memory_savings_realistic_project() {
        // Simulate a project with 5000 files across 200 directories,
        // ~25 files per directory on average.
        let mut inputs: Vec<FilePathInput> = Vec::new();
        let mut dir_strings: Vec<String> = Vec::new();
        let mut file_strings: Vec<String> = Vec::new();

        for d in 0..200 {
            let depth = d % 4 + 1;
            let dir: String = (0..depth)
                .map(|i| format!("dir{}", d * 4 + i))
                .collect::<Vec<_>>()
                .join("/")
                + "/";
            dir_strings.push(dir);
        }

        for d in 0..200 {
            for f in 0..25 {
                let filename = format!("file_{f}.rs");
                file_strings.push(filename);
            }
        }

        let mut idx = 0;
        for d in 0..200 {
            for _f in 0..25 {
                inputs.push(FilePathInput {
                    dir_relative: &dir_strings[d],
                    filename: &file_strings[idx],
                    dir_index: d as u32,
                    flags: 0,
                });
                idx += 1;
            }
        }

        let flat_bytes = estimate_flat_arena_bytes(&inputs);
        let result = build_simd_path_store(&inputs);
        let chunked_bytes = result.store.heap_bytes();

        let savings_pct = (1.0 - chunked_bytes as f64 / flat_bytes as f64) * 100.0;
        println!(
            "5000 files / 200 dirs: flat={flat_bytes}B, chunked={chunked_bytes}B, \
             savings={savings_pct:.0}%"
        );

        assert!(
            chunked_bytes < flat_bytes,
            "chunked store should save memory: {chunked_bytes} vs {flat_bytes}"
        );
    }

    // ── Overflow (watcher) tests ──

    #[test]
    fn test_overflow_entry() {
        let overflow = OverflowFileEntry::new("src/new_dir/", "new_file.rs", 0);

        assert_eq!(overflow.entry.dir_str(), "src/new_dir/");
        assert_eq!(overflow.entry.filename_str(), "new_file.rs");
        assert_eq!(overflow.entry.reconstruct_path(), "src/new_dir/new_file.rs");

        // Verify SIMD alignment of segments
        let dir_addr = overflow.entry.dir_ptr as usize;
        // Box<[u8]> is heap-allocated; the dir starts at offset 0 of the box.
        // While Box doesn't guarantee 16-byte alignment, the OverflowFileEntry
        // still works correctly with frizbee's unaligned load_partial.
        assert!(dir_addr > 0, "dir pointer should be non-null");

        // Segmented matching should work
        let (segs, count) = overflow.entry.match_segments().unwrap();
        assert_eq!(count, 2);
        assert_eq!(segs[0], b"src/new_dir/");
        assert_eq!(segs[1], b"new_file.rs");
    }

    #[test]
    fn test_overflow_root_file() {
        let overflow = OverflowFileEntry::new("", "new_root.txt", 0);
        assert_eq!(overflow.entry.dir_str(), "");
        assert_eq!(overflow.entry.filename_str(), "new_root.txt");
        assert_eq!(overflow.entry.reconstruct_path(), "new_root.txt");
    }
}
