import argparse
import logging
import os
import ssl
import socket
import sys
import time
import tempfile
from pathlib import Path
import shutil
import json

# Audio related imports
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# Local imports - change from relative to absolute imports
from src.carbonsense.core.embedding_generator import EmbeddingGenerator
from src.carbonsense.config.config_manager import ConfigManager
from src.carbonsense.utils.logger import setup_logger
from src.carbonsense.services.milvus_service import MilvusService
from src.carbonsense.core.carbon_agent import CarbonAgent
from src.carbonsense.core.crew_agent import CrewAgentManager

# Set up logger
logger = setup_logger(__name__)

# --- Constants for Recording ---
SAMPLE_RATE = 44100  # Sample rate in Hz
CHANNELS = 1 # Mono recording
# --- End Constants ---

def select_input_device_interactively() -> int | None:
    """Lists available input devices and prompts the user to select one."""
    print("\nAvailable Input Devices:")
    print("="*25)
    try:
        devices = sd.query_devices()
        input_devices = [(i, d) for i, d in enumerate(devices) if d['max_input_channels'] > 0]

        if not input_devices:
            logger.error(" No input devices found.")
            return None

        for i, (idx, device) in enumerate(input_devices):
            default_marker = " (default)" if idx == sd.default.device[0] else ""
            print(f"  {i+1}: {device['name']}{default_marker} (Index: {idx})")

        while True:
            try:
                choice = input(f"\nEnter the number of the microphone to use (1-{len(input_devices)}): ")
                choice_index = int(choice) - 1
                if 0 <= choice_index < len(input_devices):
                    selected_device_index = input_devices[choice_index][0]
                    logger.info(f"Selected device: {input_devices[choice_index][1]['name']} (Index: {selected_device_index})")
                    return selected_device_index
                else:
                    print(f"Invalid choice. Please enter a number between 1 and {len(input_devices)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except EOFError: # Handle case where input stream is closed (e.g., piping)
                 logger.error("Input stream closed. Cannot select device interactively.")
                 return None
            except KeyboardInterrupt:
                logger.info("\nDevice selection cancelled by user.")
                return None

    except Exception as e:
        logger.error(f" Could not query or select audio devices: {e}")
        return None

def fetch_certificates(config: ConfigManager) -> None:
    """Fetch and save Milvus certificates."""
    try:
        milvus_config = config.get_milvus_config()
        host = milvus_config.get('host')
        port = int(milvus_config.get('port', 30902))  # default port if missing
        
        # Use root directory for certificate
        cert_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cert_file = os.path.join(cert_path, "milvus-grpc.crt")
        
        # Backup existing certificate if it exists
        if os.path.exists(cert_file):
            backup_file = os.path.join(cert_path, f"milvus-grpc_backup_{int(time.time())}.crt")
            shutil.copy2(cert_file, backup_file)
            logger.info(f" Existing certificate backed up to: {backup_file}")
        
        logger.info(f"Fetching certificate from {host}:{port}...")
        
        try:
            # Attempt to fetch server certificate
            cert = ssl.get_server_certificate((host, port))
            
            # Write the certificate to a file
            with open(cert_file, "w") as f:
                f.write(cert)
            
            logger.info(f" Certificate successfully saved to: {cert_file}")
            
            # Update environment variable
            os.environ['MILVUS_CERT_PATH'] = cert_path
            logger.info(f"Certificate path updated to {cert_path}")
            
        except socket.gaierror as e:
            logger.error(f" DNS resolution failed for host: {host}")
            logger.error(f"Error: {e}")
            raise
            
        except Exception as e:
            logger.error(f" Failed to fetch certificate from {host}:{port}")
            logger.error(f"Error: {e}")
            raise
            
    except Exception as e:
        logger.error(f"Error in fetch_certificates: {str(e)}")
        raise

def record_audio(duration: int, sample_rate: int, channels: int, device_index: int = None) -> str:
    """Records audio from the microphone for a specified duration and saves it to a temporary WAV file."""
    try:
        device_info = f" using device index {device_index}" if device_index is not None else " using default device"
        num_frames = int(duration * sample_rate)
        logger.info(f" Attempting to record for {duration} seconds ({num_frames} frames) at {sample_rate} Hz{device_info}... Speak now!")

        start_time = time.time()
        recording = sd.rec(num_frames, samplerate=sample_rate, channels=channels, dtype='int16', device=device_index)
        sd.wait()  # Wait until recording is finished
        end_time = time.time()

        actual_duration = end_time - start_time
        logger.info(f" Recording finished. Actual duration: {actual_duration:.2f} seconds.")

        # Check if actual duration significantly deviates from requested duration
        if abs(actual_duration - duration) > 2: # Allow a 2-second buffer
             logger.warning(f" Recording duration ({actual_duration:.2f}s) significantly differs from requested ({duration}s).")

        # Create a temporary file to save the recording
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_file_path = temp_file.name
        temp_file.close() # Close the file handle so wav.write can open it

        # Save as WAV file using scipy
        wav.write(temp_file_path, sample_rate, recording)
        logger.info(f" Audio saved temporarily to: {temp_file_path}")
        return temp_file_path

    except Exception as e:
        logger.error(f" Error during audio recording: {str(e)}")
        # Attempt to clean up if file was created
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
             try:
                 os.remove(temp_file_path)
                 logger.info(f" Cleaned up temporary file: {temp_file_path}")
             except OSError as rm_err:
                 logger.error(f" Error cleaning up temp file {temp_file_path}: {rm_err}")
        raise

def transcribe_audio(config: ConfigManager, audio_file_path: str) -> str:
    """Transcribes audio using IBM Watson Speech to Text."""
    stt_config = config.get_stt_config()
    api_key = stt_config.get("api_key")
    service_url = stt_config.get("url")

    if not api_key or not service_url:
        logger.error(" IBM Speech to Text API key or URL not found in configuration.")
        raise ValueError("IBM STT configuration missing.")

    authenticator = IAMAuthenticator(api_key)
    speech_to_text = SpeechToTextV1(
        authenticator=authenticator
    )
    speech_to_text.set_service_url(service_url)

    try:
        with open(audio_file_path, 'rb') as audio_file:
            response = speech_to_text.recognize(
                audio=audio_file,
                content_type='audio/wav', # Adjust content type based on your audio file format
                model='en-US_BroadbandModel' # Specify the appropriate model
            ).get_result()
        
        if response['results']:
            transcript = response['results'][0]['alternatives'][0]['transcript']
            logger.info(" Transcription successful.")
            return transcript
        else:
            logger.warning(" No transcription results returned.")
            return ""
            
    except Exception as e:
        logger.error(f" Error during transcription: {str(e)}")
        raise
    finally:
        # Clean up the temporary audio file after transcription attempt
        if os.path.exists(audio_file_path) and tempfile.gettempdir() in os.path.dirname(audio_file_path):
            try:
                os.remove(audio_file_path)
                logger.info(f"🧹 Cleaned up temporary file: {audio_file_path}")
            except OSError as e:
                logger.error(f" Error cleaning up temp file {audio_file_path}: {str(e)}")

def process_stt_query(config: ConfigManager, audio_file_path: str, show_context: bool = False) -> dict:
    """Process a speech query using transcription and CrewAgentManager.
    
    Args:
        config: Configuration manager instance
        audio_file_path: Path to the audio file
        show_context: Whether to show context in results
        
    Returns:
        Dictionary containing the response and metadata
    """
    try:
        # Transcribe the audio
        transcript = transcribe_audio(config, audio_file_path)
        if not transcript:
            return {
                "error": "Could not transcribe audio or transcription is empty.",
                "response": "Failed to transcribe audio. Please try again."
            }
            
        print(f"\nTranscribed Query: {transcript}")
        
        # Process query using CrewAgentManager
        crew_manager = CrewAgentManager(
            config=config,
            use_cache=True,  # Enable caching for better performance
            use_hierarchical=True,  # Use hierarchical processing for complex queries
            debug_mode=False  # Set to True for detailed logs
        )
        
        return crew_manager.process_query(transcript, show_context)
        
    except Exception as e:
        logger.error(f"Error processing STT query: {str(e)}", exc_info=True)
        return {
            "error": f"Failed to process speech query: {str(e)}",
            "response": "Sorry, I encountered an error while processing your speech query."
        }

def process_stt_crew_query(config: ConfigManager, audio_file_path: str, show_context: bool = False, 
                          debug_mode: bool = False, use_hierarchical: bool = True, 
                          store_thoughts: bool = False) -> dict:
    """Process a speech query using transcription and CrewAgentManager with full CrewAI features.
    
    Args:
        config: Configuration manager instance
        audio_file_path: Path to the audio file
        show_context: Whether to show context in results
        debug_mode: Enable detailed agent interaction logs
        use_hierarchical: Use hierarchical processing instead of sequential
        store_thoughts: Store agent thoughts in log files
        
    Returns:
        Dictionary containing the response and metadata
    """
    try:
        # Transcribe the audio
        transcript = transcribe_audio(config, audio_file_path)
        if not transcript:
            return {
                "error": "Could not transcribe audio or transcription is empty.",
                "response": "Failed to transcribe audio. Please try again.",
                "transcription": ""
            }
            
        print(f"\nTranscribed Query: {transcript}")
        
        # Process query using CrewAgentManager with full feature set
        crew_manager = CrewAgentManager(
            config=config,
            debug_mode=debug_mode,
            use_hierarchical=use_hierarchical,
            store_thoughts=store_thoughts
        )
        
        print(f"\nProcessing query with CrewAI using hierarchical as {use_hierarchical}...")
        print("=" * 80)
        
        # Process the query and return results
        result = crew_manager.process_query(transcript, show_context)
        
        # Add the transcript to the result
        result["transcription"] = transcript
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing STT CrewAI query: {str(e)}", exc_info=True)
        return {
            "error": f"Failed to process speech query: {str(e)}",
            "response": "Sorry, I encountered an error while processing your speech query.",
            "transcription": ""
        }

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="CarbonSense - Carbon Footprint Analysis")
    parser.add_argument("--mode", choices=["generate", "rag_agent", "crew_agent", "verify", "cleanup", "fetch_certs", "stt_query", "stt_crew_agent"], required=True, help="Operation mode")
    parser.add_argument("--model", choices=["125m", "granite", "30m"], help="Model to use for embeddings (used in generate/verify)")
    parser.add_argument("--query", help="Query string for rag_agent and crew_agent modes")
    parser.add_argument("--record_duration", type=int, default=10, help="Duration of audio recording in seconds for stt_query and stt_crew_agent modes (default: 10)")
    parser.add_argument("--input_device", type=int, default=None, help="Index of the input audio device (optional). If not provided, you will be prompted to select.")
    parser.add_argument("--show_context", action="store_true", help="Show context in query results")
    parser.add_argument("--files", nargs="+", help="Specific files to process (used in generate mode)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no_cache", action="store_true", help="Disable caching")
    parser.add_argument("--sequential", action="store_true", help="Use sequential process instead of hierarchical")
    parser.add_argument("--store_thoughts", action="store_true", help="Store agent thoughts and reasoning in log files")
    parser.add_argument("--output_file", help="Path to a file where the results should be saved (in JSON format)")
    args = parser.parse_args()

    try:
        # Initialize config manager
        config = ConfigManager()
        
        if args.mode == "fetch_certs":
            fetch_certificates(config)
            return
            
        if args.mode == "generate":
            generator = EmbeddingGenerator(config)
            # Set model flags based on argument
            use_125m = args.model == "125m"
            use_granite = args.model == "granite"
            generator.process_all_files(use_125m, use_granite, args.files)
            
        elif args.mode == "rag_agent":
            if not args.query:
                print("Error: --query argument is required for query mode")
                return
            
            try:
                agent = CarbonAgent(config)
                result = agent.process_query(args.query)
                
                if "error" in result:
                    print(f"Error: {result['error']}")
                    return
                
                print("\nResponse:")
                print("=" * 80)
                print(result["response"])
                
                if args.show_context:
                    print("\nSources:")
                    print("-" * 40)
                    for source in result["sources"]:
                        print(f"- {source}")
                
                print(f"\nConfidence Score: {result['confidence']:.2f}")
                
            except Exception as e:
                print(f"Error processing query: {str(e)}")
                
        elif args.mode == "crew_agent":
            if not args.query:
                print("Error: --query argument is required for crew_agent mode")
                return
                
            try:
                # Initialize the CrewAgentManager with LiteLLM always enabled
                crew_manager = CrewAgentManager(
                    config=config,
                    debug_mode=args.debug,
                    use_cache=args.no_cache,
                    use_hierarchical=not args.sequential,
                    store_thoughts=args.store_thoughts
                )
                
                print(f"\nProcessing query with CrewAI using hierarchical as {not args.sequential}...")
                print("=" * 80)
                
                # Process the query using the CrewAgentManager
                result = crew_manager.process_query(args.query, args.show_context)
                
                if "error" in result:
                    print(f"Error: {result['error']}")
                    return
                
                print("\nResponse:")
                print("=" * 80)
                print(result["response"])
                
                if args.show_context and result.get("context"):
                    print("\nSources:")
                    print("-" * 40)
                    if result["context"].get("sources"):
                        for source in result["context"]["sources"]:
                            print(f"- {source}")
                    else:
                        print("No specific sources identified.")
                
            except Exception as e:
                print(f"Error processing query with CrewAI: {str(e)}")
                logger.error(f"Error in crew_agent mode: {str(e)}", exc_info=True)

        elif args.mode == "stt_query":
            selected_device_index = args.input_device
            if selected_device_index is None:
                selected_device_index = select_input_device_interactively()
                if selected_device_index is None:
                    print("Device selection failed or was cancelled. Exiting.")
                    return

            try:
                # Record audio
                temp_audio_path = record_audio(args.record_duration, SAMPLE_RATE, CHANNELS, 
                                        device_index=selected_device_index)
                
                # Process the query using CrewAgentManager
                result = process_stt_query(config, temp_audio_path, args.show_context)
                
                if "error" in result:
                    print(f"Error: {result['error']}")
                    return
                    
                print("\nResponse:")
                print("=" * 80)
                print(result["response"])
                
                if args.show_context and result.get("context"):
                    print("\nSources:")
                    print("-" * 40)
                    if result["context"].get("sources"):
                        for source in result["context"]["sources"]:
                            print(f"- {source}")
                
            except Exception as e:
                print(f"Error processing STT query: {str(e)}")

        elif args.mode == "stt_crew_agent":
            selected_device_index = args.input_device
            if selected_device_index is None:
                selected_device_index = select_input_device_interactively()
                if selected_device_index is None:
                    print("Device selection failed or was cancelled. Exiting.")
                    return

            try:
                # Record audio
                print(f"\nRecording audio for {args.record_duration} seconds...")
                temp_audio_path = record_audio(args.record_duration, SAMPLE_RATE, CHANNELS, 
                                               device_index=selected_device_index)
                
                # Process the query using CrewAgentManager with full feature set
                print("\nTranscribing and processing with CrewAI agents...")
                result = process_stt_crew_query(
                    config, 
                    temp_audio_path, 
                    show_context=args.show_context,
                    debug_mode=args.debug,
                    use_hierarchical=not args.sequential,
                    store_thoughts=args.store_thoughts
                )
                
                if "error" in result:
                    print(f"Error: {result['error']}")
                    
                    # If output file is specified, save error result to the file
                    if args.output_file:
                        try:
                            with open(args.output_file, 'w') as f:
                                json.dump(result, f)
                            print(f"Error results saved to {args.output_file}")
                        except Exception as save_err:
                            print(f"Error saving results to file: {str(save_err)}")
                            # Ensure there's at least an empty valid JSON
                            with open(args.output_file, 'w') as f:
                                json.dump({"error": "Failed to save results", "transcription": "", "response": "", "confidence": 0.0}, f)
                    
                    return
                    
                print("\nResponse:")
                print("=" * 80)
                print(result["response"])
                
                if args.show_context and result.get("context"):
                    print("\nSources:")
                    print("-" * 40)
                    if result["context"].get("sources"):
                        for source in result["context"]["sources"]:
                            print(f"- {source}")
                    else:
                        print("No specific sources identified.")
                
                # If output file is specified, save the result to the file
                if args.output_file:
                    try:
                        with open(args.output_file, 'w') as f:
                            json.dump(result, f)
                        print(f"Results saved to {args.output_file}")
                    except Exception as save_err:
                        print(f"Error saving results to file: {str(save_err)}")
                
            except Exception as e:
                print(f"Error processing speech query with CrewAI: {str(e)}")
                logger.error(f"Error in stt_crew_agent mode: {str(e)}", exc_info=True)
                
                # If output file is specified, save error to the file
                if args.output_file:
                    try:
                        error_result = {
                            "error": f"Failed to process speech query: {str(e)}",
                            "response": "Sorry, I encountered an error while processing your speech query.",
                            "transcription": ""
                        }
                        with open(args.output_file, 'w') as f:
                            json.dump(error_result, f)
                        print(f"Error saved to {args.output_file}")
                    except Exception as save_err:
                        print(f"Error saving error to file: {str(save_err)}")
                        # Last resort - ensure there's at least an empty valid JSON
                        try:
                            with open(args.output_file, 'w') as f:
                                json.dump({"error": "System error", "transcription": "", "response": "", "confidence": 0.0}, f)
                        except:
                            pass  # Nothing more we can do

        elif args.mode == "verify":
            milvus = MilvusService(config)
            
            # Define all model collections
            collections = {
                "30m": "carbon_embeddings_30m",
                "125m": "carbon_embeddings_125m",
                "granite": "carbon_embeddings_granite"
            }
            
            # If model is specified, only verify that model
            if args.model:
                if args.model not in collections:
                    logger.error(f"Invalid model: {args.model}. Must be one of: {', '.join(collections.keys())}")
                    return
                collections = {args.model: collections[args.model]}
            
            for model_name, collection_name in collections.items():
                logger.info(f"Verifying {model_name} model collection:")
                logger.info("=" * 80)
                
                stats = milvus.verify_collection(collection_name)
                
                if "error" in stats:
                    logger.error(f"Error: {stats['error']}")
                else:
                    logger.info(f"Total vectors: {stats['num_entities']}")
                    logger.info(f"Unique files: {len(stats['unique_files'])}")
                    
                    if stats['unique_files']:
                        logger.info("Files in collection:")
                        for file in stats['unique_files']:
                            logger.info(f"- {file}")
                    else:
                        logger.info("No files found in collection")
                    
        elif args.mode == "cleanup":
            try:
                from src.carbonsense.services.cleanup_service import CleanupService
                cleanup_service = CleanupService(config)
                
                print("\nStarting cleanup process...")
                print("=" * 80)
                
                # Use the cleanup_all method which handles everything properly
                cleanup_service.cleanup_all()
                
                print("\nCleanup completed successfully")
                
            except Exception as e:
                logger.error(f"Cleanup failed: {str(e)}")
                sys.exit(1)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()