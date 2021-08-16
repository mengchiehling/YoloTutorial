def allowed_file(filename: str) -> bool:

    """
    Check if the input file has the correct format

    Args:
        filename: the filename of the uploaded image

    Returns:
        if the filename has the right extension.
    """

    allowed_extensions = {'png', 'jpg', 'jpeg'}

    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions