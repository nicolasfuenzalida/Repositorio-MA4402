from pathlib import Path

def get_project_root():
    """Entrega el 'path' del proyecto.
    
    Returns
    ----------
    Path
        objeto Path con el 'path' del proyecto.
    """        
    return Path(__file__).parent.parent