# Federated CIFAR-10 with TensorFlow and MPI

This project implements both simulated and distributed federated learning for CIFAR-10 image classification using:

- **TensorFlow** for model training
- **TensorFlow Federated (TFF)** for simulation
- **MPI (via mpi4py)** for real-world client-server federation
- **Matplotlib** for training curve visualization

---

## 🚀 Setup

### 1. Install dependencies

Create a virtual environment (optional but recommended):

\`\`\`bash
python -m venv venv
source venv/bin/activate  # On Windows: .\\venv\\Scripts\\activate
\`\`\`

Install packages:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

---

## ⚙️ Run centralized training (baseline)

\`\`\`bash
python main.py
\`\`\`

---

## 🤝 Run federated simulation (TensorFlow Federated)

\`\`\`bash
python federated/main_federated.py
\`\`\`

---

## 🔁 Run federated learning with MPI (local)

Use 1 server and N clients:

\`\`\`bash
mpiexec -n 5 python federated_mpi/mpi_main.py
\`\`\`

This will:
- Split CIFAR-10 among clients
- Train 1 round per client per epoch
- Aggregate weights on the server
- Plot evaluation curves

---

## 📁 Project Structure

\`\`\`
.
├── data_loader.py
├── model.py
├── train.py
├── main.py
├── requirements.txt
├── federated/
│   ├── main_federated.py
│   ├── client.py
│   ├── server.py
├── federated_mpi/
│   ├── mpi_main.py
│   ├── mpi_client.py
│   ├── mpi_server.py
│   ├── mpi_utils.py
\`\`\`

---

## 📊 Output

At the end of training, plots of **test accuracy** and **loss** across federated rounds are displayed.

---

## 🧠 Notes

- `.npy` versions of CIFAR-10 are used to support cluster execution (no internet needed)
- On Cedar or other HPC, replace \`mpiexec\` with \`srun\` or \`mpirun\` as appropriate
- TensorFlow logs and warnings about oneDNN can be ignored

EOF
