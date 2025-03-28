import pytest
import torch
import numpy as np
import time
from pathlib import Path
from src.data.module import MPIDataModule, MPIDataset


class TestMPIDataLoader:
    @pytest.fixture
    def datamodule(self):
        return MPIDataModule(
            batch_size=2, num_workers=2, normalize=True, include_coordinates=True
        )

    def test_data_shapes(self, datamodule):
        """Test if the data shapes are correct"""
        datamodule.setup()
        batch = next(iter(datamodule.train_dataloader()))

        # Test input shapes
        inputs, mmr, cell_area = batch
        assert inputs.dim() == 5, "Input should be 5-dimensional"
        assert inputs.shape[0] == 2, "Batch size should be 2"
        assert inputs.shape[1] == 7, "Should have 7 input channels"
        assert all(x > 0 for x in inputs.shape), "All dimensions should be positive"

        # Test output shapes
        assert mmr.dim() == 5, "MMR should be 5-dimensional"
        assert mmr.shape[0] == 2, "Batch size should be 2"
        assert cell_area.dim() == 3, "Cell area should be 3-dimensional"

    def test_data_types(self, datamodule):
        """Test if the data types are correct"""
        datamodule.setup()
        batch = next(iter(datamodule.train_dataloader()))
        inputs, mmr, cell_area = batch

        assert inputs.dtype == torch.float32, "Inputs should be float32"
        assert mmr.dtype == torch.float32, "MMR should be float32"
        assert cell_area.dtype == torch.float32, "Cell area should be float32"

    def test_data_ranges(self, datamodule):
        """Test if normalized data is roughly in expected ranges"""
        datamodule.setup()
        batch = next(iter(datamodule.train_dataloader()))
        inputs, mmr, cell_area = batch

        # For normalized data, most values should be within [-5, 5]
        assert torch.abs(inputs).mean() < 5, "Normalized inputs seem out of range"
        assert torch.abs(mmr).mean() < 5, "Normalized MMR seem out of range"
        assert torch.all(cell_area > 0), "Cell areas should be positive"

    def test_iteration_speed(self, datamodule):
        """Test if data iteration is reasonably fast"""
        datamodule.setup()
        train_loader = datamodule.train_dataloader()

        start_time = time.time()
        num_batches = 0
        batch_times = []

        for batch in train_loader:
            batch_start = time.time()
            # Simulate some processing
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
            num_batches += 1

        total_time = time.time() - start_time
        avg_batch_time = np.mean(batch_times)

        # Assert reasonable timing (adjust thresholds as needed)
        assert (
            avg_batch_time < 1.0
        ), f"Average batch processing too slow: {avg_batch_time:.2f}s"
        assert (
            total_time < num_batches * 1.0
        ), f"Total iteration too slow: {total_time:.2f}s"

    def test_memory_usage(self, datamodule):
        """Test if memory usage is reasonable"""
        import psutil

        process = psutil.Process()

        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        datamodule.setup()
        train_loader = datamodule.train_dataloader()

        # Load a few batches
        for i, batch in enumerate(train_loader):
            if i >= 3:  # Test with first 3 batches
                break

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (adjust threshold as needed)
        assert memory_increase < 1000, f"Memory usage too high: {memory_increase:.2f}MB"

    def test_data_splits(self, datamodule):
        """Test if data splits are correct"""
        datamodule.setup()

        train_size = len(datamodule.train_dataset)
        val_size = len(datamodule.val_dataset)
        test_size = len(datamodule.test_dataset)
        total_size = train_size + val_size + test_size

        # Check if splits roughly match the specified ratios
        assert abs(train_size / total_size - 0.5) < 0.01, "Train split incorrect"
        assert abs(val_size / total_size - 0.25) < 0.01, "Validation split incorrect"
        assert abs(test_size / total_size - 0.25) < 0.01, "Test split incorrect"

    def test_reproducibility(self, datamodule):
        """Test if data loading is reproducible with same seed"""
        from src.utils import set_seed

        set_seed(42)
        datamodule.setup()
        batch1 = next(iter(datamodule.train_dataloader()))

        set_seed(42)
        datamodule.setup()
        batch2 = next(iter(datamodule.train_dataloader()))

        # Compare tensors
        assert torch.allclose(batch1[0], batch2[0]), "Inputs not reproducible"
        assert torch.allclose(batch1[1], batch2[1]), "MMR not reproducible"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_transfer(self, datamodule):
        """Test if data can be moved to GPU efficiently"""
        datamodule.setup()
        batch = next(iter(datamodule.train_dataloader()))

        start_time = time.time()
        inputs, mmr, cell_area = [x.cuda() for x in batch]
        torch.cuda.synchronize()
        transfer_time = time.time() - start_time

        assert transfer_time < 1.0, f"GPU transfer too slow: {transfer_time:.2f}s"
        assert inputs.is_cuda and mmr.is_cuda and cell_area.is_cuda, "Data not on GPU"
