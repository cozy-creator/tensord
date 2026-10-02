//! CPU metadata/source gate against an explicitly selected existing snapshot.
//! `--verify` additionally admits a copied closure into an owned target's native catalog.
#[allow(dead_code)]
#[path = "../model_sources.rs"]
mod model_sources;
#[allow(dead_code)]
#[path = "../os.rs"]
mod os;

use model_sources::{ModelSources, SelectedManifest, SourceRequest, SourceRole};
use std::{
    fs::File,
    io::{self, Read},
    path::PathBuf,
};
use tensorfs_core::{
    header::Header,
    read::{self, Source},
    store::Store,
};

fn main() -> io::Result<()> {
    let mut args = std::env::args().skip(1);
    let first = args
        .next()
        .ok_or_else(|| io::Error::other("store root required"))?;
    let verify = first == "--verify";
    let root = PathBuf::from(
        if verify { args.next() } else { Some(first) }
            .ok_or_else(|| io::Error::other("store root required"))?,
    );
    let manifest = args
        .next()
        .ok_or_else(|| io::Error::other("manifest required"))?;
    let component_list = args
        .next()
        .ok_or_else(|| io::Error::other("component required"))?;
    let components: Vec<String> = component_list.split(',').map(str::to_owned).collect();
    let output = PathBuf::from(
        args.next()
            .ok_or_else(|| io::Error::other("output directory required"))?,
    );
    if args.next().is_some() {
        return Err(io::Error::other("unexpected argument"));
    }
    if verify {
        Store::ensure(&root).map_err(io::Error::other)?;
    }
    std::fs::create_dir_all(&output)?;
    let mut broker = ModelSources::open(
        &root,
        &[SelectedManifest {
            manifest: manifest.clone(),
            components: components.clone(),
        }],
    )?;
    if verify {
        let (objects, bytes) = broker.verify_selected()?;
        println!("verified selected source objects={objects} bytes={bytes}");
    }
    let paths = broker.closure_paths(&manifest)?;
    let relative: Vec<_> = paths
        .iter()
        .map(|path| path.strip_prefix(&root).unwrap().display().to_string())
        .collect();
    std::fs::write(output.join("closure-files.txt"), relative.join("\n") + "\n")?;
    let mut grant = broker.read(&SourceRequest {
        manifest: manifest.clone(),
        role: SourceRole::Header,
        name: String::new(),
        length: 0,
    })?;
    let mut bytes = Vec::new();
    grant.file.read_to_end(&mut bytes)?;
    let header = Header::parse(&bytes).map_err(io::Error::other)?;
    let mut traversal = Vec::new();
    for component in &components {
        let tensors = header
            .components
            .iter()
            .find(|(name, _)| name == component)
            .unwrap();
        traversal.extend(
            tensors
                .1
                .iter()
                .map(|(key, _)| (component.clone(), key.clone())),
        );
    }
    let tensor_count = traversal.len();
    let plan = read::plan_for_traversal(&header, &traversal, &components, 4 << 20)
        .map_err(io::Error::other)?;
    let object = plan
        .items
        .iter()
        .find_map(|item| match &item.source {
            Source::Object(range) => Some(range.obj.clone()),
            _ => None,
        })
        .unwrap();
    let mut body = broker.read(&SourceRequest {
        manifest,
        role: SourceRole::Object,
        name: object.id(),
        length: object.length,
    })?;
    let destination = output.join("selected-object");
    let mut file = File::create(&destination)?;
    let copied = io::copy(&mut body.file, &mut file)?;
    file.sync_all()?;
    let mut permissions = file.metadata()?.permissions();
    permissions.set_readonly(true);
    file.set_permissions(permissions)?;
    let metadata = serde_json::json!({"header_sha256": grant.sha256, "header_length": grant.length,
        "tensor_count": tensor_count, "plan_items": plan.items.len(),
        "object_sha256": object.sha256, "object_length": object.length, "copied": copied});
    std::fs::write(output.join("header.canonical"), bytes)?;
    std::fs::write(
        output.join("metadata.json"),
        serde_json::to_vec_pretty(&metadata).unwrap(),
    )?;
    println!("{}", metadata);
    Ok(())
}
