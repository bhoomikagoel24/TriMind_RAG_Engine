import sys
from typing import Optional
from TriMind_RAG_Engine.logging.logger import get_logger

logger = get_logger(__name__)

class CustomException(Exception):
    """
    Custom Exception class for detailed error tracking.
    """

    def __init__(self, error_message: str, error_detail: Optional[sys] = None):
        super().__init__(error_message)
        self.error_message = self._get_detailed_error(error_message, error_detail)

    def _get_detailed_error(self, error_message, error_detail: Optional[sys]):
        if error_detail:
            _, _, exc_tb = error_detail.exc_info()

            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno

            detailed_message = (
                f"Error in file: {file_name} "
                f"at line: {line_number} | "
                f"Message: {error_message}"
            )

            logger.error(detailed_message)
            return detailed_message

        return error_message

    def __str__(self):
        return self.error_message
