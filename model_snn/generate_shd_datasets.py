from shd_exp.datasets import BinnedSpikingHeidelbergDigits

# Generate SHD datasets with specific configurations
def generate_datasets(duration=50, n_bins=4, frames=None, split_by=None):
    """
    Generate training and testing datasets for SHD.

    Parameters:
        duration (int): Duration of the dataset.
        n_bins (int): Number of bins for spiking data.
        frames (int or None): Number of frames for data splitting.
        split_by (str or None): Criterion for splitting data.

    Returns:
        tuple: Training and testing datasets.
    """
    train_dataset = BinnedSpikingHeidelbergDigits(
        './data/SHD', n_bins=n_bins, train=True, data_type='frame', 
        frames_number=frames, split_by=split_by, duration=duration
    )
    test_dataset = BinnedSpikingHeidelbergDigits(
        './data/SHD', n_bins=n_bins, train=False, data_type='frame', 
        frames_number=frames, split_by=split_by, duration=duration
    )
    return train_dataset, test_dataset

if __name__ == "__main__":
    # Example usage
    train_dataset, test_dataset = generate_datasets(duration=50, n_bins=4, frames=None, split_by=None)