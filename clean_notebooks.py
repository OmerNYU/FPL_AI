import nbformat
import os

def clean_notebook(notebook_path):
    """Clean a Jupyter notebook by removing Colab badges and metadata."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Remove Colab metadata
    if 'colab' in nb.metadata:
        del nb.metadata['colab']
    
    # Clean cells
    cleaned_cells = []
    for cell in nb.cells:
        # Skip cells that are Colab badges
        if cell.cell_type == 'markdown' and 'colab-badge.svg' in cell.source:
            continue
        
        # Clean cell metadata
        if 'colab' in cell.metadata:
            del cell.metadata['colab']
        
        cleaned_cells.append(cell)
    
    nb.cells = cleaned_cells
    
    # Write back the cleaned notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def main():
    # List of notebooks to clean
    notebooks = [
        'train_model.ipynb',
        'weekly_fixtures.ipynb',
        'weekly_results.ipynb',
        'weekly_performance.ipynb',
        'clean_previous_seasons.ipynb',
        'clean_fixtures.ipynb'
    ]
    
    for notebook in notebooks:
        if os.path.exists(notebook):
            print(f"Cleaning {notebook}...")
            clean_notebook(notebook)
            print(f"Done cleaning {notebook}")
        else:
            print(f"Warning: {notebook} not found")

if __name__ == "__main__":
    main() 