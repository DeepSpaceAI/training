import os

def most_recent_checkpoint(checkpointing_folder) -> str:
    """
    This function produces the path to the most recently saved checkpoint by
    taking the largest checkpoint iteration number. It converts that number
    to a filer path and returns the file path.
    """

    # Checks to see if a given checkpointing folder even exists
    if not os.path.exists(checkpointing_folder):
        raise Exception("Path does not exist")
    
    # Go into the checkpointing folder and retrieve the set of checkpoint names.
    folders = next(iter(os.walk(checkpointing_folder)))[1]

    # Checks to see if there are checkpoints to be loaded.
    if len(folders) == 0:
        raise Exception("There are no checkpoints in the folder")

    # Parse the iteration numbers and store them in the list
    checkpoint_nums = list()
    for folder in folders:
        checkpoint_nums.append(int(folder.split("_")[1]))
    
    # Retrieve the largest iteration number
    most_recent_checkpoint = str(sorted(checkpoint_nums)[-1])

    # Return a path with the given checkpoint folder.
    return os.path.join(checkpointing_folder, f"checkpoint_{most_recent_checkpoint}")
    