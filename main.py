"""Drive applicaiton."""

from commands   import *
from utilities  import ARGS, BANNER, LOGGER

if __name__ == "__main__":
    """Execute application command(s)."""
    
    try:# Log banner
        LOGGER.info(BANNER)
        
        print(vars(ARGS))
        
        # Dtermine command being executed.
        match ARGS.command:
            
            # Download dataset.
            case "download-dataset":    download_dataset(**vars(ARGS))
            
            # Run job process.
            case "run-job":             run_job(**vars(ARGS))
            
            # Invalid command.
            case _:                     LOGGER.error(f"Invalid command provided: {ARGS.command}")
        
    # Gracefully handle keyboard interruptions
    except KeyboardInterrupt:   LOGGER.info("Keyboard interruption detected. Aborting operations.")
        
    # Catch wildcard errors
    except Exception as e:      LOGGER.error(f"Unexpected error: {e}")
    
    # Exit gracefully
    finally:                    LOGGER.info("Exiting...")