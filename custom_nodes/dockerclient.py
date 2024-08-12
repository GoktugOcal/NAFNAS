import json
import logging
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Dict, List, Tuple

from core.emulator.distributed import DistributedServer
from core.errors import CoreCommandError, CoreError
from core.executables import BASH
from core.nodes.base import CoreNode, CoreNodeOptions

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from core.emulator.session import Session

DOCKER: str = "docker"

from core.nodes.docker import DockerNode, DockerOptions, DockerVolume


class DockerClient(DockerNode):
    """
    Docker based node with resource limits (CPU or memory).
    """

    def __init__(
        self,
        session: "Session",
        _id: int = None,
        name: str = None,
        server: DistributedServer = None,
        options: DockerOptions = None,
    ) -> None:
        """
        Create a DockerNode instance.

        :param session: core session instance
        :param _id: node id
        :param name: node name
        :param server: remote server node
            will run on, default is None for localhost
        :param options: options for creating node
        """
        super().__init__(session, _id, name, server, options)
        self.no_cpus = options.no_cpus
        self.mem_limit = options.mem_limit

    
    def startup(self) -> None:
        """
        Create a docker container instance for the specified image.

        :return: nothing
        """
        with self.lock:
            if self.up:
                raise CoreError(f"starting node({self.name}) that is already up")
            # create node directory
            self.makenodedir()
            # setup commands for creating bind/volume mounts
            binds = ""
            for src, dst in self.binds:
                binds += f"--mount type=bind,source={src},target={dst} "
            volumes = ""
            for volume in self.volumes.values():
                volumes += (
                    f"--mount type=volume," f"source={volume.src},target={volume.dst} "
                )
            # normalize hostname
            hostname = self.name.replace("_", "-")
            # create container and retrieve the created containers PID
            self.host_cmd(
                f"{DOCKER} run -td --init --net=none --hostname {hostname} "
                f"--cpus=\"{self.no_cpus}\" "
                f"--memory=\"{self.mem_limit}\" "
                f"--name {self.name} --sysctl net.ipv6.conf.all.disable_ipv6=0 "
                f"{binds} {volumes} "
                f"--privileged {self.image} tail -f /dev/null"
            )
            # retrieve pid and process environment for use in nsenter commands
            self.pid = self.host_cmd(
                f"{DOCKER} inspect -f '{{{{.State.Pid}}}}' {self.name}"
            )
            output = self.host_cmd(f"cat /proc/{self.pid}/environ")
            for line in output.split("\x00"):
                if not line:
                    continue
                key, value = line.split("=")
                self.env[key] = value
            # setup symlinks for bind and volume mounts within
            for src, dst in self.binds:
                link_path = self.host_path(Path(dst), True)
                self.host_cmd(f"ln -s {src} {link_path}")
            for volume in self.volumes.values():
                volume.path = self.host_cmd(
                    f"{DOCKER} volume inspect -f '{{{{.Mountpoint}}}}' {volume.src}"
                )
                link_path = self.host_path(Path(volume.dst), True)
                self.host_cmd(f"ln -s {volume.path} {link_path}")
            logger.debug("node(%s) pid: %s", self.name, self.pid)
            self.up = True
