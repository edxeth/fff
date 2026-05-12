use std::io::Write;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use fff_search::file_picker::{FilePicker, FilePickerOptions};
use fff_search::grep::{GrepSearchOptions, GrepMode, parse_grep_query};

/// Create a temporary directory with realistic source files for benchmarking.
fn setup_repo() -> (tempfile::TempDir, FilePicker) {
    let dir = tempfile::tempdir().unwrap();

    for i in 0..10 {
        let mut f = std::fs::File::create(dir.path().join(format!("file_{}.rs", i))).unwrap();
        for j in 0..1000 {
            writeln!(
                f,
                "pub fn function_{}_{}(arg: {}) -> {} {{",
                i, j,
                ["String", "u32", "bool", "Vec<u8>", "Option<String>"][j % 5],
                ["Result<()>", "usize", "String", "bool", "Vec<u8>"][j % 5]
            )
            .unwrap();
            writeln!(f, "    let x = {}_{};", i, j).unwrap();
            writeln!(
                f,
                "    let y = validate_{}_{};",
                ["input", "output", "schema", "config", "result"][j % 5],
                j
            )
            .unwrap();
            writeln!(f, "    Ok(x)").unwrap();
            writeln!(f, "}}").unwrap();
            if j % 10 == 0 {
                writeln!(
                    f,
                    "// TODO: validate_input_{} and validate_output_{} together",
                    j, j
                )
                .unwrap();
            }
        }
    }

    let mut picker = FilePicker::new(FilePickerOptions {
        base_path: dir.path().to_str().unwrap().into(),
        watch: false,
        ..Default::default()
    })
    .unwrap();
    picker.collect_files().unwrap();

    (dir, picker)
}

fn grep_opts() -> GrepSearchOptions {
    GrepSearchOptions {
        max_file_size: 10 * 1024 * 1024,
        max_matches_per_file: 0,
        smart_case: true,
        file_offset: 0,
        page_limit: 100,
        mode: GrepMode::PlainText,
        time_budget_ms: 0,
        before_context: 0,
        after_context: 0,
        classify_definitions: false,
        trim_whitespace: false,
        abort_signal: None,
    }
}

fn bench_multi_grep_vs_sequential(c: &mut Criterion) {
    let (_dir, picker) = setup_repo();
    let opts = grep_opts();

    let mut group = c.benchmark_group("multi_grep_vs_sequential");

    // 2-pattern case
    group.bench_function("multi_grep/2_patterns", |b| {
        b.iter(|| {
            black_box(picker.multi_grep(
                &["validate_input", "validate_output"],
                &[],
                &opts,
            ))
        })
    });

    group.bench_function("2x_grep_sequential/2_patterns", |b| {
        let q1 = parse_grep_query("validate_input");
        let q2 = parse_grep_query("validate_output");
        b.iter(|| {
            black_box(picker.grep(&q1, &opts));
            black_box(picker.grep(&q2, &opts));
        })
    });

    // 5-pattern case
    let five_patterns = [
        "validate_input",
        "validate_output",
        "validate_schema",
        "validate_config",
        "validate_result",
    ];

    group.bench_function("multi_grep/5_patterns", |b| {
        b.iter(|| {
            black_box(picker.multi_grep(
                &five_patterns,
                &[],
                &opts,
            ))
        })
    });

    group.bench_function("5x_grep_sequential/5_patterns", |b| {
        let queries: Vec<_> = five_patterns.iter().map(|p| parse_grep_query(p)).collect();
        b.iter(|| {
            for q in &queries {
                black_box(picker.grep(q, &opts));
            }
        })
    });

    group.finish();
}

criterion_group!(benches, bench_multi_grep_vs_sequential);
criterion_main!(benches);
