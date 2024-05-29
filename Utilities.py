import os
import shutil

def setup_directories(name='dataset_name'):
    # Verify if the 'figures' directory exists create it if it does not
    figures_dir = 'figures'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Same for the output_dir
    vit_debugg_dir = os.path.join(figures_dir, name)

    # Remove the output_dir if it already exists
    if os.path.exists(vit_debugg_dir):
        shutil.rmtree(vit_debugg_dir)

    # Create the 'ViT_debugg' directory
    os.makedirs(vit_debugg_dir)
    print(f'All figures will be saved in {vit_debugg_dir} folder.')
