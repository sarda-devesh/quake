import subprocess
import psutil
import os
import signal
import glob

class RunningWorker:
    """
    Class to store all of the metadata associated with a running leaviathan worker
    """
    def __init__(self, worker_address, running_proc, logfile):
        self.worker_address_ = worker_address
        self.running_proc_ = running_proc
        self.logfile_ = logfile
    
    def stop_worker(self):
        try:
            os.killpg(os.getpgid(self.running_proc_.pid), signal.SIGTERM)
        finally:
            self.logfile_.close()

class LeviathanClusterManager:

    """
    A class to start/stop/manage a single Leviathan cluster. A single leviathan cluster
    is made up of a single coordinator node, and multiple compute and storage nodes
    """
    def __init__(self, build_dir, num_compute, num_storage, base_port = 8000):
        """
        Initializes the Cluster (without starting any of the nodes) with the specified parameter
        :param build_dir: The path to the build directory with the executables for the nodes
        :param num_compute: The number of compute nodes we want to launch
        :param num_storage: The number of storage nodes we want to launch
        :param base_port: The starting port to use for generating ports for these works grpc services
        """
        self.build_dir_ = build_dir
        self.num_computes_ = num_compute
        self.num_storage_ = num_storage
        self.curr_port_ = base_port

        self.workers_by_type = { 
            "compute_node" : [],
            "storage_node" : [],
            "coordinator" : [] 
        }

        self.perform_cleanup(True)
    
    def delete_cluster_data(self, delete_log_files = False):
        """
        Deletes any data created by the cluster
        """
        start_dir = os.getcwd()
        os.chdir(self.build_dir_)

        files_to_delete_regex = ["partition_*"]
        if delete_log_files:
            files_to_delete_regex.append("*.log")

        try:
            files_to_remove = []
            for pattern in files_to_delete_regex:
                files_to_remove.extend(glob.glob(pattern))
                
            for file in files_to_remove:
                os.remove(file)
        except OSError as e:
            print(f"Could not delete {file}: {e}")

        os.chdir(start_dir)
    
    def perform_cleanup(self, delete_log_files = False): 
        """
        Cleans up any resources from any previous experiments
        """

        # Stop any running workers
        expected_executable_names = ["./" + node_type for node_type in self.workers_by_type.keys()]
        for proc in psutil.process_iter(['pid', 'cmdline']):
            try:
                if proc.info['cmdline'] is None or proc.pid is None:
                    continue
                cmdline = " ".join(proc.info['cmdline'])
                if any(node in cmdline for node in expected_executable_names):
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Also cleanup any old partition files that were not deleted
        self.delete_cluster_data(delete_log_files=delete_log_files)
    
    def launch_worker_of_type(self, worker_type):
        """
        Launches worker of the specified type
        """
        # First build the excutable for the worker type
        build_command = f"make {worker_type} -j$(nproc)"
        subprocess.check_output(build_command, shell = True)

        # Determine the number of workers
        num_workers = 1
        if worker_type == "compute_node":
            num_workers = self.num_computes_
        elif worker_type == "storage_node":
            num_workers = self.num_storage_

        # Launch that many workers
        workers_arr = self.workers_by_type[worker_type]
        for _ in range(num_workers):
            # Determine worker metadata
            curr_port = self.curr_port_
            self.curr_port_ += 1
            logfile_name = f"{worker_type}_{curr_port}.log"

            # Determine command to run
            if worker_type == "coordinator":
                command_to_run = f"./{worker_type} --port={curr_port}"
            else:
                coordinator_address = self.workers_by_type["coordinator"][0].worker_address_
                command_to_run = f"./{worker_type} --port={curr_port} --coordinator_address={coordinator_address}"
            print("Running worker", command_to_run, "whose output is logged to", logfile_name)

            # Launch the worker
            logfile = open(logfile_name, "w+")
            proc = subprocess.Popen(
                command_to_run, shell=True, preexec_fn=os.setsid,
                stdout=logfile, stderr=logfile
            )
            running_worker = RunningWorker(f"localhost:{curr_port}", proc, logfile)
            workers_arr.append(running_worker)
    
    def start_cluster(self):
        """
        Starts up the cluster by launching the specified number of workers
        """
        start_dir = os.getcwd()
        os.chdir(self.build_dir_)

        try:
            self.launch_worker_of_type("coordinator")
            self.launch_worker_of_type("storage_node")
            self.launch_worker_of_type("compute_node")
            return True
        except Exception as error:
            print("[ERROR] Failed to launch cluster due to error", error)
        finally:
            os.chdir(start_dir)

        return False
    
    def get_compute_node_address(self):
        return [worker.worker_address_ for worker in self.workers_by_type["compute_node"]]
    
    def stop_cluster(self):
        """
        Stops all of the clusters that were started up
        """
        for worker_type in self.workers_by_type:
            for curr_worker in self.workers_by_type[worker_type]:
                curr_worker.stop_worker()
        
        self.delete_cluster_data(False)