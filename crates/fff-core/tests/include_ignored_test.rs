use fff_search::file_picker::{FilePicker, FilePickerOptions};

#[test]
fn test_include_ignored_flag() {
    let dir = tempfile::tempdir().unwrap();
    let dp = dir.path();

    std::process::Command::new("git").args(["init"]).current_dir(dp).output().unwrap();
    std::fs::write(dp.join("tracked.txt"), b"data\n").unwrap();
    std::fs::write(dp.join("ignored.txt"), b"data\n").unwrap();
    std::fs::write(dp.join(".gitignore"), b"ignored.txt\n").unwrap();
    std::process::Command::new("git").args(["add", "-A"]).current_dir(dp).output().unwrap();
    std::process::Command::new("git").args(["commit", "-m", "init"]).current_dir(dp).output().unwrap();

    // Without include_ignored
    let mut p1 = FilePicker::new(FilePickerOptions {
        base_path: dp.to_str().unwrap().into(),
        include_ignored: false, watch: false, ..Default::default()
    }).unwrap();
    p1.collect_files().unwrap();
    let has1 = p1.get_files().iter().any(|f| f.relative_path(&p1).ends_with("ignored.txt"));
    assert!(!has1, "include_ignored=false should exclude ignored.txt");

    // With include_ignored
    let mut p2 = FilePicker::new(FilePickerOptions {
        base_path: dp.to_str().unwrap().into(),
        include_ignored: true, watch: false, ..Default::default()
    }).unwrap();
    p2.collect_files().unwrap();
    let has2 = p2.get_files().iter().any(|f| f.relative_path(&p2).ends_with("ignored.txt"));
    assert!(has2, "include_ignored=true should find ignored.txt");
}
