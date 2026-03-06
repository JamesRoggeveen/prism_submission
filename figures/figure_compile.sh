#!/bin/bash

# Set the source directory where .tex files are located
source_dir="figure_latex"

# Set the destination directory where .tex files will be copied
temp_dir="figure_code_temp"

# Set the regex pattern to search for
search_pattern='\\includegraphics(.*)\{figures'

# Set the replacement string
replacement='\\includegraphics\1{..'

# Create the destination directory if it doesn't exist
mkdir -p "$temp_dir"

# Copy .tex files to the destination directory
cp "$source_dir"/fig_*.tex "$temp_dir"

# Iterate over each .tex file in the destination directory
for tex_file in "$temp_dir"/*.tex; do
    # Use sed to find and replace the regex pattern in each file
    sed -i.bak -E "s/$search_pattern/$replacement/g" "$tex_file"

    # Remove the backup file created by sed (optional)
    rm -f "${tex_file}.bak"
done

echo "Copy and replace operation completed."

# Set the destination directory where .pdf files will be moved
destination_dir="../finished_figures"


# Navigate to the source directory
cd "$temp_dir" || exit

# Compile each .tex file in the source directory
for tex_file in *.tex; do
    # Check if the file exists and is not a directory
    if [ -f "$tex_file" ]; then
        # Compile the .tex file to generate the corresponding .pdf
        pdflatex -interaction=batchmode "$tex_file"
        
        # Extract the filename without extension
        filename=$(basename -- "$tex_file")
        filename_no_ext="${filename%.*}"

        # Move the resulting .pdf file to the destination directory
        mv "$filename_no_ext.pdf" "$destination_dir"
    fi
done

cd - || exit

# Clean up helper files in the source directory
rm -f *.log *.aux *.toc *.out
rm -rf $temp_dir

rm -f $source_dir/*.log
rm -f $source_dir/*.aux
rm -f $source_dir/*.toc
rm -f $source_dir/*.gz
rm -f $source_dir/*.fls
rm -f $source_dir/*.fdb_latexmk
rm -f $source_dir/*.pdf

echo "Compilation and cleanup completed."