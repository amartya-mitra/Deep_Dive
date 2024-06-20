import posixpath
from typing import Any, Callable

from file_system import FileSystem, FileSystemEvent, FileSystemEventType

# If you're completing this task in an online assessment, you can increment this
# constant to enable more unit tests relevant to the task you are on (1-5).
TASK_NUM = 1

class ReplicatorSource:
    """Class representing the source side of a file replicator."""

    def __init__(self, fs: FileSystem, dir_path: str, rpc_handle: Callable[[Any], Any]):
        self._fs = fs
        self._dir_path = dir_path
        self._rpc_handle = rpc_handle

        # Start watching the directory for changes
        self._fs.watchdir(self._dir_path, self.handle_event)


    def handle_event(self, event: FileSystemEvent):
        """Handle a file system event.

        Used as the callback provided to FileSystem.watchdir().
        """
        # Construct the request based on the event type

        request = {
            'event_type': event.event_type,
            'src_path': self._dir_path,
            'dest_path': event.path
        }

        # Send the request to the target through the RPC handle
        self._rpc_handle(request)

class ReplicatorTarget:
    """Class representing the target side of a file replicator."""

    def __init__(self, fs: FileSystem, dir_path: str):
        self._fs = fs
        self._dir_path = dir_path

    def handle_request(self, request: Any) -> Any:
        """Handle a request from the ReplicatorSource."""

        event_type = request['event_type']
        src_path = request['src_path']
        dest_path = request.get('dest_path', '')

        # Map source path to target path
        relative_path = posixpath.relpath(src_path, self._dir_path)
        target_path = posixpath.join(self._dir_path, relative_path)

        print(src_path, self._dir_path, self._fs.exists("./"))


        if event_type == FileSystemEventType.FILE_OR_SUBDIR_ADDED:
            # Create the file or directory
            if self._fs.isdir(src_path):
                self._fs.mkdir(target_path)
            else:
                self._fs.writefile(target_path, self._fs.readfile(src_path))

        elif event_type == FileSystemEventType.FILE_OR_SUBDIR_REMOVED:

            print(self._fs.exists('./base'))
            # Delete the file or directory
            if self._fs.isdir(src_path):
                self._fs.rmdir(target_path)
            else:
                self._fs.remove(target_path)

        elif event_type == FileSystemEventType.FILE_MODIFIED:
            # Modify the file
            self._fs.writefile(target_path, self._fs.readfile(src_path))

        return {'status': 'success'}