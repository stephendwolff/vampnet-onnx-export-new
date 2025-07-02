import torch
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from pathlib import Path
import time

from audiotools import AudioSignal

# from dac.model.dac import DAC
from lac.model.lac import LAC as DAC


class LACModelValidator:
    def __init__(self, pytorch_model, encoder_onnx_path, decoder_onnx_path):
        """
        Initialize validator with PyTorch model and ONNX model paths

        Args:
            pytorch_model: Your original LAC PyTorch model
            encoder_onnx_path: Path to exported encoder ONNX file
            decoder_onnx_path: Path to exported decoder ONNX file
        """
        self.pytorch_model = pytorch_model
        self.pytorch_model.eval()

        # Load ONNX models
        self.encoder_session = ort.InferenceSession(encoder_onnx_path)
        self.decoder_session = ort.InferenceSession(decoder_onnx_path)

        # Get input/output names
        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        self.encoder_output_name = self.encoder_session.get_outputs()[0].name
        self.decoder_input_name = self.decoder_session.get_inputs()[0].name
        self.decoder_output_name = self.decoder_session.get_outputs()[0].name

        print("✓ ONNX models loaded successfully")
        print(f"Encoder input: {self.encoder_input_name}")
        print(f"Encoder output: {self.encoder_output_name}")
        print(f"Decoder input: {self.decoder_input_name}")
        print(f"Decoder output: {self.decoder_output_name}")

    def generate_test_audio(self, sample_rate=44100, duration=2.0, audio_type="sine"):
        """Generate test audio signals"""
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples)

        if audio_type == "sine":
            # Simple sine wave
            audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        elif audio_type == "chirp":
            # Frequency sweep
            f0, f1 = 100, 2000
            audio = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / duration) * t)
        elif audio_type == "noise":
            # White noise
            audio = np.random.randn(samples) * 0.1
        elif audio_type == "mixed":
            # Mix of sine waves
            audio = (
                np.sin(2 * np.pi * 440 * t)
                + 0.5 * np.sin(2 * np.pi * 880 * t)
                + 0.25 * np.sin(2 * np.pi * 1320 * t)
            )

        # Normalize and add batch/channel dimensions
        audio = audio.astype(np.float32)
        audio = audio / (np.abs(audio).max() + 1e-8)  # Normalize
        audio = audio.reshape(1, 1, -1)  # (batch, channels, samples)

        return torch.from_numpy(audio)

    def run_pytorch_inference(self, audio_tensor):
        """Run inference through PyTorch model"""
        with torch.no_grad():
            # Handle different model structures
            if hasattr(self.pytorch_model, "encoder") and hasattr(
                self.pytorch_model, "decoder"
            ):
                # Direct encoder/decoder access
                encoded = self.pytorch_model.encoder(audio_tensor)
                decoded = self.pytorch_model.decoder(encoded)
            elif hasattr(self.pytorch_model, "_orig_mod"):
                # Wrapped model
                encoded = self.pytorch_model._orig_mod.encoder(audio_tensor)
                decoded = self.pytorch_model._orig_mod.decoder(encoded)
            else:
                # Try full model
                decoded = self.pytorch_model(audio_tensor)
                encoded = None

        return encoded, decoded

    def run_onnx_inference(self, audio_tensor):
        """Run inference through ONNX models"""
        audio_numpy = audio_tensor.numpy()

        # Encoder
        encoder_output = self.encoder_session.run(
            [self.encoder_output_name], {self.encoder_input_name: audio_numpy}
        )[0]

        # Decoder
        decoder_output = self.decoder_session.run(
            [self.decoder_output_name], {self.decoder_input_name: encoder_output}
        )[0]

        return torch.from_numpy(encoder_output), torch.from_numpy(decoder_output)

    def compare_outputs(self, pytorch_output, onnx_output, name=""):
        """Compare PyTorch and ONNX outputs"""
        if pytorch_output is None or onnx_output is None:
            print(f"⚠️  Cannot compare {name}: One output is None")
            return

        # Convert to numpy for comparison
        if isinstance(pytorch_output, torch.Tensor):
            pytorch_np = pytorch_output.numpy()
        else:
            pytorch_np = pytorch_output

        if isinstance(onnx_output, torch.Tensor):
            onnx_np = onnx_output.numpy()
        else:
            onnx_np = onnx_output

        # Calculate metrics
        abs_diff = np.abs(pytorch_np - onnx_np)
        rel_diff = abs_diff / (np.abs(pytorch_np) + 1e-8)

        max_abs_diff = abs_diff.max()
        mean_abs_diff = abs_diff.mean()
        max_rel_diff = rel_diff.max()
        mean_rel_diff = rel_diff.mean()

        # Calculate correlation
        correlation = np.corrcoef(pytorch_np.flatten(), onnx_np.flatten())[0, 1]

        print(f"\n{name} Comparison:")
        print(f"  Shape: PyTorch {pytorch_np.shape}, ONNX {onnx_np.shape}")
        print(f"  Max absolute difference: {max_abs_diff:.2e}")
        print(f"  Mean absolute difference: {mean_abs_diff:.2e}")
        print(f"  Max relative difference: {max_rel_diff:.2e}")
        print(f"  Mean relative difference: {mean_rel_diff:.2e}")
        print(f"  Correlation: {correlation:.6f}")

        # Quality assessment
        if max_abs_diff < 1e-5:
            print(f"  ✓ Excellent match")
        elif max_abs_diff < 1e-3:
            print(f"  ✓ Good match")
        elif max_abs_diff < 1e-1:
            print(f"  ⚠️  Moderate differences")
        else:
            print(f"  ❌ Large differences")

        return {
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "max_rel_diff": max_rel_diff,
            "mean_rel_diff": mean_rel_diff,
            "correlation": correlation,
        }

    def benchmark_performance(self, audio_tensor, num_runs=10):
        """Benchmark inference speed"""
        print(f"\nPerformance Benchmark ({num_runs} runs):")

        # PyTorch timing
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()

        for _ in range(num_runs):
            with torch.no_grad():
                _, pytorch_decoded = self.run_pytorch_inference(audio_tensor)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        pytorch_time = (time.time() - start_time) / num_runs

        # ONNX timing
        start_time = time.time()

        for _ in range(num_runs):
            _, onnx_decoded = self.run_onnx_inference(audio_tensor)

        onnx_time = (time.time() - start_time) / num_runs

        print(f"  PyTorch: {pytorch_time*1000:.2f} ms/inference")
        print(f"  ONNX: {onnx_time*1000:.2f} ms/inference")
        print(f"  Speedup: {pytorch_time/onnx_time:.2f}x")

        return pytorch_time, onnx_time

    def visualize_comparison(self, audio_tensor, save_plots=True):
        """Create visualization comparing outputs - BETTER LINE STYLES"""
        # Run inference
        pytorch_encoded, pytorch_decoded = self.run_pytorch_inference(audio_tensor)
        onnx_encoded, onnx_decoded = self.run_onnx_inference(audio_tensor)

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Convert ALL to numpy arrays first
        original = audio_tensor[0, 0].numpy()
        pytorch_recon = pytorch_decoded[0, 0].numpy()
        onnx_recon = onnx_decoded[0, 0]

        # Ensure onnx_recon is numpy
        if hasattr(onnx_recon, "numpy"):
            onnx_recon = onnx_recon.numpy()

        # Trim all to shortest length
        min_length = min(len(original), len(pytorch_recon), len(onnx_recon))
        original_trimmed = original[:min_length]
        pytorch_recon_trimmed = pytorch_recon[:min_length]
        onnx_recon_trimmed = onnx_recon[:min_length]

        print(
            f"Visualization using {min_length} samples (trimmed from {len(original)})"
        )

        time_axis = np.arange(min_length) / 44100
        plot_samples = min(1000, min_length)

        # Plot waveforms with distinct styles
        axes[0, 0].plot(
            time_axis[:plot_samples],
            original_trimmed[:plot_samples],
            label="Original",
            color="blue",
            linewidth=2,
            alpha=0.8,
        )
        axes[0, 0].plot(
            time_axis[:plot_samples],
            pytorch_recon_trimmed[:plot_samples],
            label="PyTorch",
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
        )
        axes[0, 0].plot(
            time_axis[:plot_samples],
            onnx_recon_trimmed[:plot_samples],
            label="ONNX",
            color="green",
            linestyle=":",
            linewidth=2,
            alpha=0.8,
        )
        axes[0, 0].set_title(f"Audio Waveforms (first {plot_samples} samples)")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Amplitude")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Error plots with distinct styles
        pytorch_diff = np.abs(original_trimmed - pytorch_recon_trimmed)
        onnx_diff = np.abs(original_trimmed - onnx_recon_trimmed)

        axes[0, 1].plot(
            time_axis[:plot_samples],
            pytorch_diff[:plot_samples],
            label="PyTorch Error",
            color="red",
            linestyle="-",
            linewidth=2,
            alpha=0.8,
        )
        axes[0, 1].plot(
            time_axis[:plot_samples],
            onnx_diff[:plot_samples],
            label="ONNX Error",
            color="green",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
        )
        axes[0, 1].set_title("Reconstruction Error")
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("Absolute Error")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Encoded features comparison with distinct styles
        if pytorch_encoded is not None and onnx_encoded is not None:
            if hasattr(pytorch_encoded, "numpy"):
                pytorch_enc = pytorch_encoded[0, :, 0].numpy()
            else:
                pytorch_enc = pytorch_encoded[0, :, 0]

            if hasattr(onnx_encoded, "numpy"):
                onnx_enc = onnx_encoded[0, :, 0].numpy()
            else:
                onnx_enc = onnx_encoded[0, :, 0]

            axes[1, 0].plot(
                pytorch_enc,
                label="PyTorch",
                color="red",
                linestyle="-",
                linewidth=1.5,
                alpha=0.8,
            )
            axes[1, 0].plot(
                onnx_enc,
                label="ONNX",
                color="green",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
            )
            axes[1, 0].set_title("Encoded Features (first time frame)")
            axes[1, 0].set_xlabel("Feature Index")
            axes[1, 0].set_ylabel("Value")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Feature difference
            enc_diff = np.abs(pytorch_enc - onnx_enc)
            axes[1, 1].plot(enc_diff, color="purple", linewidth=1.5)
            axes[1, 1].set_title("Encoded Feature Differences")
            axes[1, 1].set_xlabel("Feature Index")
            axes[1, 1].set_ylabel("Absolute Difference")
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Length comparison bars
            lengths = [len(original), len(pytorch_recon), len(onnx_recon)]
            labels = ["Original", "PyTorch", "ONNX"]
            colors = ["blue", "red", "green"]

            bars1 = axes[1, 0].bar(labels, lengths, color=colors, alpha=0.7)
            axes[1, 0].set_title("Output Lengths")
            axes[1, 0].set_ylabel("Samples")
            axes[1, 0].grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, length in zip(bars1, lengths):
                height = bar.get_height()
                axes[1, 0].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{length}",
                    ha="center",
                    va="bottom",
                )

            # Trimming info
            trimmed = [
                len(original) - min_length,
                len(pytorch_recon) - min_length,
                len(onnx_recon) - min_length,
            ]

            bars2 = axes[1, 1].bar(labels, trimmed, color=colors, alpha=0.7)
            axes[1, 1].set_title("Samples Trimmed for Comparison")
            axes[1, 1].set_ylabel("Trimmed Samples")
            axes[1, 1].grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, trim in zip(bars2, trimmed):
                height = bar.get_height()
                if height > 0:
                    axes[1, 1].text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{trim}",
                        ha="center",
                        va="bottom",
                    )

        plt.tight_layout()

        if save_plots:
            plt.savefig("lac_comparison.png", dpi=300, bbox_inches="tight")
            print("✓ Comparison plot saved as 'lac_comparison.png'")

        plt.show()

    def validate_multiple_inputs(self, test_types=["sine", "chirp", "noise", "mixed"]):
        """Validate with multiple input types"""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE VALIDATION")
        print("=" * 60)

        all_results = {}

        for test_type in test_types:
            print(f"\n--- Testing with {test_type} audio ---")

            # Generate test audio
            test_audio = self.generate_test_audio(audio_type=test_type)
            AudioSignal(test_audio, sample_rate=44100).write(
                f"original_{test_type}_output.wav"
            )

            # Run inference
            pytorch_encoded, pytorch_decoded = self.run_pytorch_inference(test_audio)
            onnx_encoded, onnx_decoded = self.run_onnx_inference(test_audio)
            AudioSignal(pytorch_decoded, sample_rate=44100).write(
                f"pytorch_{test_type}_output.wav"
            )
            AudioSignal(onnx_decoded, sample_rate=44100).write(
                f"onnx_{test_type}_output.wav"
            )

            # AudioSignal(torch.from_numpy(onnx_decoded), sample_rate=44100).write(
            #     f"onnx_{test_type}_output.wav"
            # )

            # Compare results
            encoded_results = self.compare_outputs(
                pytorch_encoded, onnx_encoded, "Encoded Features"
            )
            decoded_results = self.compare_outputs(
                pytorch_decoded, onnx_decoded, "Reconstructed Audio"
            )

            all_results[test_type] = {
                "encoded": encoded_results,
                "decoded": decoded_results,
            }

        return all_results

    def run_full_validation(self, skip_validation=False):
        """Run complete validation suite"""
        print("Starting LAC Model Validation...")

        # Generate test audio
        test_audio = self.generate_test_audio(audio_type="mixed")

        # Validate multiple inputs
        results = self.validate_multiple_inputs()

        # Performance benchmark
        self.benchmark_performance(test_audio)

        # Create visualizations
        try:
            self.visualize_comparison(test_audio)
        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")
            print("Continuing without plots...")
        # Summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        for test_type, test_results in results.items():
            if test_results["decoded"]:
                corr = test_results["decoded"]["correlation"]
                max_diff = test_results["decoded"]["max_abs_diff"]
                print(
                    f"{test_type:10}: Correlation={corr:.6f}, Max Diff={max_diff:.2e}"
                )

        return results


# Usage example:
def main():

    codec_ckpt = str((Path(__file__).parent / "../models/vampnet/codec.pth").resolve())
    device = "cpu"
    lac_model = DAC.load(codec_ckpt)
    lac_model.to(device)
    lac_model.eval()

    # Load your PyTorch model

    # Create validator
    lac_encoder_onnx_path = (Path(__file__).parent / "../lac_encoder.onnx").resolve()
    lac_decoder_onnx_path = (Path(__file__).parent / "../lac_decoder.onnx").resolve()
    validator = LACModelValidator(
        pytorch_model=lac_model,
        encoder_onnx_path=lac_encoder_onnx_path,  # "lac_encoder.onnx",
        decoder_onnx_path=lac_decoder_onnx_path,  # "lac_decoder.onnx",
    )

    # Run validation
    results = validator.run_full_validation()

    return results


if __name__ == "__main__":
    results = main()
