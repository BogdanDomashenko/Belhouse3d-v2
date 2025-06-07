# BelHouse3D — Fixed Version

This is a **maintained and partially fixed version** of the [original BelHouse3D project](https://github.com/umaatgithub/BelHouse3D?tab=readme-ov-file), with contributions and fixes by **Bohdan Domashenko**. In this version, only **PointNet** has been fixed and tested. Other models may still have issues and require further debugging.

**Related dataset backup repository**: [Backup-Touchstone3D](https://github.com/umaatgithub/Backup-Touchstone3D)

---

## ⚙️ Requirements

You need a **powerful PC with an NVIDIA GPU**, or you can run it on the **KU Leuven HPC cluster** using [VSC OnDemand](https://ondemand.hpc.kuleuven.be).

To get access:

1. Apply for a VSC account via your project curator.
2. Once approved, go to [https://ondemand.hpc.kuleuven.be/pun/sys/dashboard](https://ondemand.hpc.kuleuven.be/pun/sys/dashboard) and use the **Login Server (Shell Access)**.

---

## 📁 Environment Setup (VSC)

1. **Navigate to your data directory**:

   ```bash
   cd $VSC_DATA
   ```

2. **Create a directory for your Python packages**:

   ```bash
   mkdir python_packages
   ```

3. **Install required packages into this directory**:

   ```bash
   pip install -r requirements.txt --target=$VSC_DATA/python_packages
   ```

4. **Clone this repository**:

   ```bash
   git clone https://github.com/your-fork/belhouse3d.git
   ```

---

## 🧠 Dataset Setup

1. **Download and extract the dataset** from KU Leuven RDR:
   [https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/ZS8D6K](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/ZS8D6K)

2. **Place the dataset under this structure** inside your `$VSC_DATA` directory:

   ```
   data/
   └── belhouse3d/
       ├── raw/
       │   └── IID-nonoccluded/
       │       ├── House1
       │       ├── House2
       │       └── ...
       └── processed/
           ├── train/
           │   ├── rooms/
           │   └── blocks/
           ├── val/
           │   ├── rooms/
           │   └── blocks/
           ├── test/
           │   ├── rooms/
           │   └── blocks/
           └── meta/
   ```

   > ⚠️ You’ll get the `processed/` folder **after running** the dataset processing script.

3. **(Optional)**: You may need to **reduce the dataset** to fit resource limits. If you do so, **update the following config files** accordingly:

   - `benchmark_semseg/cfg/pointnet_c.cfg`
   - `benchmark_semseg/cfg/pointnet_c_test.cfg`

---

## 🛠️ Step-by-Step: Running the Project

### 1. Process the Dataset

Edit the `process_belhouse3d_iid.yaml` file with the correct paths, then run:

```bash
./process_data_semseg.sh
```

---

### 2. Start an Interactive GPU Job (VSC)

From [the dashboard](https://ondemand.hpc.kuleuven.be/pun/sys/dashboard), go to:

**Interactive Apps → Interactive Shell → Launch with the following settings:**

- **Cluster**: `genius`
- **Partition**: `gpu_v100`
- **Account**: `lp_bel_house_3d`
- **Number of hours**: `4`
- **Nodes**: `1`
- **Tasks per node**: `1`
- **Cores per task**: `1`
- **Memory per core (MB)**: `12000`
- **GPUs**: `1`

Once launched, run the following to load your environment:

```bash
module load cluster/genius/gpu_v100
module load Python/3.11.3-GCCcore-12.3.0
export PYTHONPATH=$VSC_DATA/python_packages
```

---

### 3. Train the Model

Update paths in:

- `train_sem_seg.py`
- `test_sem_seg.py`

Then navigate to the repo and run:

```bash
python train_sem_seg.py --cfg ./benchmark_semseg/cfg/pointnet_c.cfg
```

---

### 4. Test the Model

Once training is complete, run:

```bash
python test_sem_seg.py --cfg ./benchmark_semseg/cfg/pointnet_c_test.cfg
```

---

## 📌 Notes

- In this version only **PointNet** model has been fixed other models may also need some fixes and improvements.
- Make sure you carefully check all **path references** in config files.
- Dataset is large. If you reduce it, make sure your configs match the structure and class count.
