#2018 - Aishwarya

##Code Changes

* Adding the ability to add upto 10 grain sizes (minimum 3)

********************************************************
Older Version Of Hapke
# Hapke in Python

## TODO

 * add a system for showing progress messages during section runs
  + make the main / handler async
  + have the /progress handler cache messages until requested
  + have the JS poll a /progress endpoint at regular intervals until done
 * move the ProgramState object to a separate file
  + make the cli UI use ProgramState as well

## Code Structure

Entry-points to the Hapke codebase:

 * `web.py`: Starts the web-based Hapke interface.
 * `cli.py`: Runs all the Hapke code from the command line, without prompting
   the user for additional information.
 * `rc_demo.py` Simple web UI for playing with radiance coefficients.

Each of the above relies on some of these support modules:

 * `analysis.py`: Contains the meat of the Hapke code.
 * `hapke_model.py`: Provides an object representing the Hapke model,
   useful for caching intermediate values and keeping track of the moving parts.
 * `mplweb.py`: Handles the web UI plumbing, including websockets.

Additional files:

 * `ui.html`: The HTML template used for the main web UI.
 * `html/`: Static files needed for the main web UI.
