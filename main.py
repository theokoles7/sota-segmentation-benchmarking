"""Drive applicaiton."""

from commands   import *
from utilities  import *

if __name__ == "__main__":
    """Execute application command(s)."""
    
    try:# Log banner
        LOGGER.info(BANNER)
        
        # Match command.
        match ARGS.command:
            
            # Execute job.
            case "benchmark":   run_benchmark(**vars(ARGS))
        
    # Gracefully handle keyboard interruptions
    except KeyboardInterrupt:   LOGGER.info("Keyboard interruption detected. Aborting operations.")
        
    # Catch wildcard errors
    except Exception as e:      LOGGER.error(f"Unexpected error: {e}", exc_info = True)
    
    # Exit gracefully
    finally:                    LOGGER.info("Exiting...")