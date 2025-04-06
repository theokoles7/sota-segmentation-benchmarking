"""Drive applicaiton."""

from utilities  import ARGS, BANNER, LOGGER

if __name__ == "__main__":
    """Execute application command(s)."""
    
    try:# Log banner
        LOGGER.info(BANNER)
        
    # Gracefully handle keyboard interruptions
    except KeyboardInterrupt:   LOGGER.info("Keyboard interruption detected. Aborting operations.")
        
    # Catch wildcard errors
    except Exception as e:      LOGGER.error(f"Unexpected error: {e}")
    
    # Exit gracefully
    finally:                    LOGGER.info("Exiting...")