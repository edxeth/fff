//! Profiling benchmark — run under `sample`, `cargo instruments`, or standalone.
//! Usage: cargo run --release --bin bench_segmented --features mimalloc -- <repo_path> [query]

#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use fff_search::file_picker::{FFFMode, FilePicker, FilePickerOptions};
use fff_search::types::PaginationArgs;
use fff_search::{FuzzySearchOptions, QueryParser};
use std::time::Instant;

fn get_rss_mb() -> f64 {
    #[cfg(target_os = "macos")]
    {
        use std::mem::MaybeUninit;
        let mut info = MaybeUninit::<libc::mach_task_basic_info_data_t>::uninit();
        let mut count = (std::mem::size_of::<libc::mach_task_basic_info_data_t>()
            / std::mem::size_of::<libc::natural_t>()) as u32;
        let ret = unsafe {
            libc::task_info(
                libc::mach_task_self(),
                libc::MACH_TASK_BASIC_INFO,
                info.as_mut_ptr() as *mut _,
                &mut count,
            )
        };
        if ret == libc::KERN_SUCCESS {
            let info = unsafe { info.assume_init() };
            return info.resident_size as f64 / 1_048_576.0;
        }
        0.0
    }
    #[cfg(not(target_os = "macos"))]
    {
        0.0
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: bench_segmented <repo_path> [query]");
        std::process::exit(1);
    }

    let repo_path = &args[1];
    let query = args.get(2).map(|s| s.as_str()).unwrap_or("controller");

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let rss_before = get_rss_mb();

    // --- INDEX ---
    let index_start = Instant::now();
    let mut picker = FilePicker::new(FilePickerOptions {
        base_path: repo_path.to_string(),
        warmup_mmap_cache: false,
        mode: FFFMode::Neovim,
        ..Default::default()
    })
    .expect("Failed to create FilePicker");
    picker.collect_files().expect("Failed to collect files");
    let index_time = index_start.elapsed();
    let rss_after_index = get_rss_mb();

    let files = picker.get_files();
    let dirs = picker.get_dirs();

    // Struct size analysis
    let file_item_size = std::mem::size_of::<fff_search::types::FileItem>();
    let dir_item_size = std::mem::size_of::<fff_search::types::DirItem>();
    eprintln!(
        "=== {} ===",
        repo_path.rsplit('/').next().unwrap_or(repo_path)
    );
    eprintln!("Files: {}, Dirs: {}", files.len(), dirs.len());
    eprintln!(
        "FileItem size: {} bytes ({:.2} MB for {} files)",
        file_item_size,
        files.len() as f64 * file_item_size as f64 / 1_048_576.0,
        files.len()
    );
    eprintln!(
        "DirItem size: {} bytes ({:.2} MB for {} dirs)",
        dir_item_size,
        dirs.len() as f64 * dir_item_size as f64 / 1_048_576.0,
        dirs.len()
    );
    eprintln!("Index time: {:.1}ms", index_time.as_secs_f64() * 1000.0);

    // Chunked filename cost analysis: each filename would be stored as
    // 16B-aligned chunks prefixed with dir overlap bytes
    let mut total_overlap_bytes = 0usize;
    let mut total_chunked_fname_bytes = 0usize;
    for f in files.iter() {
        let dir_len = f.dir_str().len();
        let fname_len = f.file_name().len();
        let overlap = dir_len % 16; // bytes of dir in the bridge chunk
        let chunked = ((overlap + fname_len + 15) / 16) * 16; // 16B-aligned
        total_overlap_bytes += overlap;
        total_chunked_fname_bytes += chunked;
    }

    // Arena size analysis
    let total_dir_bytes: usize = files.iter().map(|f| f.dir_str().len()).sum();
    let total_fname_bytes: usize = files.iter().map(|f| f.file_name().len()).sum();
    let total_path_bytes: usize = total_dir_bytes + total_fname_bytes;
    let unique_dir_bytes: usize = dirs
        .iter()
        .map(|d| {
            let r = d.relative_path();
            if r.is_empty() {
                0
            } else {
                r.len() + 1
            }
        })
        .sum();
    let padded_dir_bytes: usize = dirs
        .iter()
        .map(|d| {
            let r = d.relative_path();
            let dir_len = if r.is_empty() { 0 } else { r.len() + 1 };
            ((dir_len + 15) / 16) * 16
        })
        .sum();
    let file_item_growth = files.len() * (file_item_size - 80);
    let (arena_chunk, arena_fname, arena_overflow) = picker.arena_bytes();
    let arena_total = arena_chunk + arena_fname + arena_overflow;
    eprintln!(
        "Path bytes: total={:.2}MB (dir={:.2}MB, fname={:.2}MB), unique_dirs={:.2}MB (padded={:.2}MB), \
         dedup_savings={:.2}MB, struct_growth={:.2}MB, net={:.2}MB",
        total_path_bytes as f64 / 1_048_576.0,
        total_dir_bytes as f64 / 1_048_576.0,
        total_fname_bytes as f64 / 1_048_576.0,
        unique_dir_bytes as f64 / 1_048_576.0,
        padded_dir_bytes as f64 / 1_048_576.0,
        (total_dir_bytes - unique_dir_bytes) as f64 / 1_048_576.0,
        file_item_growth as f64 / 1_048_576.0,
        (total_dir_bytes as isize - padded_dir_bytes as isize - file_item_growth as isize) as f64 / 1_048_576.0,
    );
    eprintln!(
        "Arenas: chunk={:.2}MB, fname={:.2}MB, total={:.2}MB (old flat would be {:.2}MB)",
        arena_chunk as f64 / 1_048_576.0,
        arena_fname as f64 / 1_048_576.0,
        arena_total as f64 / 1_048_576.0,
        total_path_bytes as f64 / 1_048_576.0,
    );
    let total_fname_raw: usize = files.iter().map(|f| f.file_name().len()).sum();
    eprintln!(
        "Chunked fname: {:.2}MB (overlap={:.2}MB, padding={:.2}MB) vs packed {:.2}MB → +{:.2}MB",
        total_chunked_fname_bytes as f64 / 1_048_576.0,
        total_overlap_bytes as f64 / 1_048_576.0,
        (total_chunked_fname_bytes - total_fname_raw - total_overlap_bytes) as f64 / 1_048_576.0,
        total_fname_raw as f64 / 1_048_576.0,
        (total_chunked_fname_bytes - total_fname_raw) as f64 / 1_048_576.0,
    );

    eprintln!(
        "RSS: {:.1} MB (delta: {:.1} MB)",
        rss_after_index,
        rss_after_index - rss_before
    );

    // --- MATCHING ---
    let parser = QueryParser::default();
    let make_opts = || FuzzySearchOptions {
        max_threads: threads,
        current_file: None,
        project_path: None,
        combo_boost_score_multiplier: 100,
        min_combo_count: 3,
        pagination: PaginationArgs {
            offset: 0,
            limit: 100,
        },
    };

    // Warmup
    let parsed = parser.parse("warmup");
    for _ in 0..5 {
        let _ = FilePicker::fuzzy_search(files, &parsed, None, make_opts(), None);
    }

    let parsed = parser.parse(query);
    eprintln!("\nQuery: '{}', {} threads, 500 iterations", query, threads);
    eprintln!("PID: {} — attach profiler now", std::process::id());

    let iterations = 500;
    let mut total = std::time::Duration::ZERO;
    let mut matches = 0;

    for _ in 0..iterations {
        let start = Instant::now();
        let sr = FilePicker::fuzzy_search(files, &parsed, None, make_opts(), None);
        total += start.elapsed();
        matches = sr.total_matched;
    }

    let avg_ms = total.as_secs_f64() * 1000.0 / iterations as f64;
    eprintln!("Avg: {:.3}ms, Matches: {}", avg_ms, matches);

    let rss_final = get_rss_mb();
    eprintln!("RSS final: {:.1} MB", rss_final);
}
