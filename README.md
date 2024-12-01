# Liver Segmentation using U-Net

This project implements a U-Net model for liver segmentation using medical images. The model is trained on a dataset of liver images and corresponding masks. The project includes model definition, training, evaluation, and visualization of predictions.

## Project Structure

- **Model**: A U-Net architecture is used for image segmentation.
- **Training**: The model is trained using PyTorch with a binary cross-entropy loss function and the Adam optimizer.
- **Evaluation**: Mean Intersection over Union (mIoU) and Mean Pixel Accuracy (mPA) metrics are computed to evaluate the model's performance.
- **Visualization**: Five sample images are displayed at the end, comparing true labels with predicted masks.

## Usage

To run the project, open the provided Jupyter notebook (`.ipynb`) and execute the cells in order. The notebook contains:

- Data loading and preprocessing.
- U-Net model definition.
- Training loop with loss tracking.
- Evaluation and metric computation.
- Visualization of results.

## Requirements

The following libraries are required to run the notebook:

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- PIL (Python Imaging Library)

You can install the dependencies using:

```bash
pip install torch torchvision numpy matplotlib scikit-learn pillow

```

## Results

The model achieves good segmentation performance on the test set. Example images with true labels and predicted masks are visualized at the end of the notebook.
