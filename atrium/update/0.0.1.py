                    # Process each Python file in the directory
                    for i, script_file in enumerate(solution_files):
                        try:
                            file_path = os.path.join(solution_entry.path, script_file)
                            metadata = extract_metadata(file_path)
                            
                            # Get cover image path consistently
                            cover_image_path = get_cover_image_path(
                                solution_entry, entry, solution_name, metadata, SITE_CONFIG
                            )
                            
                            base_url = SITE_CONFIG['base_url']
                            script_path = f"{entry.name}/{solution_name}/{script_file}"
                            
                            # Direct link to specific file in the main repository
                            github_file_url = f"{main_repo_url}/blob/{main_repo_branch}/{entry.name}/{solution_name}/{script_file}"
                            print(f"Creating GitHub source URL: {github_file_url}")
                            
                            # Create a version-specific redirect for each script
                            script_filename = os.path.splitext(script_file)[0]
                            source_redirect_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="0;url={github_file_url}">
    <title>Redirecting to GitHub Source</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; text-align: center; padding: 2rem; }}
        a {{ color: #2563eb; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>View Source on GitHub</h1>
    <p>Redirecting to <a href="{github_file_url}">source code on GitHub</a>...</p>
    <script>
        window.location.href = "{github_file_url}";
    </script>
</body>
</html>
"""
                            # Create version-specific source.html file
                            with open(os.path.join(solution_output, f"source_{script_filename}.html"), "w") as f:
                                f.write(source_redirect_html)
                            
                            # Only add version suffix for non-primary scripts
                            is_primary = (script_file == primary_script)
                            version_suffix = "" if is_primary else f" ({script_filename})"
                            
                            solution_metadata = {
                                "name": metadata.get("title", solution_name) + version_suffix,
                                "description": metadata.get("description", "No description provided."),
                                "link": f"{entry.name}/{solution_name}",
                                "cover": cover_image_path,
                                "author": metadata.get("author", ""),
                                "version": metadata.get("version", ""),
                                "external_source": metadata.get("external_source", ""),
                                "script_source": f"{base_url}/{script_path}",
                                "github_source_url": github_file_url,
                                "script_filename": script_filename,
                                "is_primary": is_primary
                            }
                            
                            # Generate solution page with consistent cover image path and GitHub source URL
                            template_vars = {
                                'title': solution_metadata["name"],
                                'project_name': SITE_CONFIG['project_name'],
                                'site_config': SITE_CONFIG,
                                'cover_image': cover_image_path,
                                'description': solution_metadata["description"],
                                'link': solution_metadata["link"],
                                'author': metadata.get("author", ""),
                                'version': metadata.get("version", ""),
                                'license': metadata.get("license", ""),
                                'dependencies': metadata.get("dependencies", []),
                                'external_source': solution_metadata["external_source"],
                                'script_source': solution_metadata["script_source"],
                                'github_source_url': solution_metadata["github_source_url"],
                                'keywords': metadata.get("keywords", []),
                                'requires_python': metadata.get("requires_python", ""),
                                'repository': metadata.get("repository", ""),
                                'cli_args': extract_typer_args(file_path),
                                'documentation': metadata.get("documentation", ""),
                                'homepage': metadata.get("homepage", "")
                            }
                            
                            # Create a unique HTML file for each script version
                            with open(os.path.join(solution_output, f"index_{script_filename}.html"), "w") as f:
                                f.write(Template(SOLUTION_TEMPLATE).render(**template_vars))
                            
                            # For the main/default index.html, use the primary script
                            if is_primary:
                                with open(os.path.join(solution_output, "index.html"), "w") as f:
                                    f.write(Template(SOLUTION_TEMPLATE).render(**template_vars))
                                with open(os.path.join(solution_output, "source.html"), "w") as f:
                                    f.write(source_redirect_html)
                            
                            # Only add the primary script to the solutions list for the main index
                            # This ensures each directory is represented by a single card
                            if is_primary:
                                solutions.append(solution_metadata)
                            
                        except Exception as e:
                            print(f"Error processing script {entry.name}/{solution_entry.name}/{script_file}: {e}")
                            continue

    # Generate index page and sitemap with direct GitHub links for each card
    try:
        print(f"Generating index with {len(solutions)} solutions")
        with open(os.path.join(static_dir, "index.html"), "w") as f:
            context = {
                'solutions': solutions,
                'site_config': SITE_CONFIG,
                'categories': list(set(s["link"].split("/")[0] for s in solutions if "link" in s and "/" in s["link"]))
            }
            f.write(Template(INDEX_TEMPLATE).render(**context))
        
        generate_sitemap_txt(solutions, static_dir)
        
        # Copy CNAME if it exists
        cname_path = "./CNAME"
        if os.path.exists(cname_path):
            shutil.copy(cname_path, static_dir)
        else:
            print(f"Warning: {cname_path} not found. Skipping CNAME copy.")
            
        # Copy icon_transparent.png if it exists
        icon_path = "./icon_transparent.png"
        if os.path.exists(icon_path):
            shutil.copy(icon_path, static_dir)
        else:
            print(f"Warning: {icon_path} not found. Site may be missing its logo.")
            
    except Exception as e:
        print(f"Error generating index or sitemap: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_static_site(BASE_DIR, STATIC_DIR)