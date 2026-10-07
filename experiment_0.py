"""
Tact AI - Experiment 0: Minimal Parameter Identifiability Test
Testing whether a learned tactile dynamics model can infer an unseen physical parameter (friction mu)
and improve multi-step rollout predictions within <= 3-5 micro-interactions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# --- 1. PHYSICAL SIMULATOR (Coulomb Contact Dynamics) ---
class ContactPhysicsSimulator:
    def __init__(self, dt=0.01, k_N=100.0, mass=0.5):
        self.dt = dt
        self.k_N = k_N  # Normal stiffness (N/m)
        self.mass = mass

    def step(self, state, action, mu):
        """
        state: [f_N, f_T, x, v]
        action: [target_depth, v_cmd]
        mu: ground truth friction coefficient
        """
        f_N_prev, f_T_prev, x_prev, v_prev = state
        target_depth, v_cmd = action

        # 1. Normal contact force (spring penetration model)
        f_N = max(0.0, self.k_N * target_depth)

        # 2. Tangential friction dynamics
        # If no contact, shear is zero
        if f_N <= 1e-4:
            f_T = 0.0
            v = v_cmd
            x = x_prev + v * self.dt
            slip = 0.0
            return np.array([f_N, f_T, x, v], dtype=np.float32)

        # Contact exists: finger attempts to move at v_cmd relative to block
        # Elastic shear buildup before slip (tangential stiffness k_T)
        k_T = 200.0
        delta_x = v_cmd * self.dt
        f_T_trial = f_T_prev + k_T * delta_x
        f_max = mu * f_N

        if abs(f_T_trial) <= f_max:
            # Sticking regime: force equals trial elastic force, no macroscopic slip
            f_T = f_T_trial
            v = 0.0
            x = x_prev
        else:
            # Slipping regime: Coulomb saturation
            f_T = f_max * np.sign(f_T_trial)
            # Acceleration of the sliding interface
            a_slip = (abs(f_T_trial) - f_max) / self.mass * np.sign(f_T_trial)
            v = v_prev + a_slip * self.dt
            x = x_prev + v * self.dt

        # Add small measurement noise
        f_N_noisy = max(0.0, f_N + np.random.normal(0, 0.02))
        f_T_noisy = f_T + np.random.normal(0, 0.02)
        return np.array([f_N_noisy, f_T_noisy, x, v], dtype=np.float32)

    def rollout(self, initial_state, actions, mu):
        states = [initial_state]
        curr = initial_state
        for a in actions:
            curr = self.step(curr, a, mu)
            states.append(curr)
        return np.array(states)


# --- 2. PARAMETER-CONDITIONED DYNAMICS MODEL ---
class ParameterConditionedDynamicsModel(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64):
        super().__init__()
        # Input: state (4) + action (2) + scalar friction estimate (1) = 7
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim)  # Predicts delta_state
        )

    def forward(self, state, action, mu_est):
        """
        state: [batch, 4]
        action: [batch, 2]
        mu_est: [batch, 1]
        """
        x = torch.cat([state, action, mu_est], dim=-1)
        delta_state = self.net(x)
        return state + delta_state


# --- 3. DATA GENERATION & MODEL TRAINING ---
def generate_training_data(sim, num_trajectories=300, traj_len=20, mu_range=(0.45, 0.55)):
    states_all, actions_all, mus_all, next_states_all = [], [], [], []
    for _ in range(num_trajectories):
        mu = np.random.uniform(*mu_range)
        init_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        curr_state = init_state
        for _ in range(traj_len):
            depth = np.random.uniform(0.01, 0.08)  # Normal penetration 1-8 cm
            v_cmd = np.random.uniform(-0.15, 0.15)  # Tangential speed
            action = np.array([depth, v_cmd], dtype=np.float32)
            next_state = sim.step(curr_state, action, mu)
            
            states_all.append(curr_state)
            actions_all.append(action)
            mus_all.append([mu])
            next_states_all.append(next_state)
            curr_state = next_state

    return (torch.tensor(np.array(states_all), dtype=torch.float32),
            torch.tensor(np.array(actions_all), dtype=torch.float32),
            torch.tensor(np.array(mus_all), dtype=torch.float32),
            torch.tensor(np.array(next_states_all), dtype=torch.float32))

def train_dynamics_model(model, states, actions, mus, next_states, epochs=60, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(states, actions, mus, next_states)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    print(f"Training Dynamics Model on {len(states)} transitions...")
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for b_s, b_a, b_mu, b_ns in loader:
            optimizer.zero_grad()
            pred_ns = model(b_s, b_a, b_mu)
            loss = loss_fn(pred_ns, b_ns)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (ep + 1) % 20 == 0:
            print(f"  Epoch {ep+1}/{epochs} - Loss: {total_loss/len(loader):.6f}")


# --- 4. EXPERIMENT 0 EVALUATION PROTOCOL ---
def run_experiment_0(sim, model):
    """
    Test whether the model can infer unseen mu* via test-time optimization
    from <= 3 micro-probe steps, and whether it improves multi-step rollouts.
    """
    model.eval()
    test_cases = [
        {"name": "Slippery Teflon (OOD Low)", "mu_true": 0.15},
        {"name": "Grippy Rubber (OOD High)", "mu_true": 0.85}
    ]
    
    results = {}
    
    for tc in test_cases:
        mu_true = tc["mu_true"]
        print(f"\n=======================================================")
        print(f"Running Test Case: {tc['name']} (Ground Truth mu* = {mu_true})")
        print(f"=======================================================")

        # Define 3 micro-probe actions: normal press + progressive shear displacement
        probe_actions = [
            np.array([0.05, 0.00], dtype=np.float32),  # Step 1: Normal press into contact
            np.array([0.05, 0.04], dtype=np.float32),  # Step 2: Micro-shear probe
            np.array([0.05, 0.08], dtype=np.float32),  # Step 3: Continued shear to induce slip
        ]

        # Execute probes in environment
        state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        probe_obs = []
        for a in probe_actions:
            next_state = sim.step(state, a, mu_true)
            probe_obs.append((state, a, next_state))
            state = next_state

        # Initial belief prior
        mu_est = torch.tensor([[0.50]], requires_grad=True, dtype=torch.float32)
        opt_mu = optim.Adam([mu_est], lr=0.08)

        mu_history = [mu_est.item()]
        print(f"Initial Belief: mu_0 = {mu_est.item():.3f}")

        # Test-time adaptation on the 3 probe transitions
        for step in range(15):  # 15 gradient steps on the 3 transitions
            opt_mu.zero_grad()
            total_loss = 0.0
            for s, a, ns in probe_obs:
                t_s = torch.tensor(s).unsqueeze(0)
                t_a = torch.tensor(a).unsqueeze(0)
                t_ns = torch.tensor(ns).unsqueeze(0)
                pred_ns = model(t_s, t_a, mu_est)
                
                # Residual loss on shear force (index 1) and slip motion
                loss = nn.MSELoss()(pred_ns[:, 1], t_ns[:, 1]) + 0.5 * nn.MSELoss()(pred_ns[:, 2:], t_ns[:, 2:])
                total_loss += loss
            
            total_loss.backward()
            opt_mu.step()
            with torch.no_grad():
                mu_est.clamp_(0.02, 1.20)
            mu_history.append(mu_est.item())

        mu_final = mu_est.item()
        error_initial = abs(0.50 - mu_true)
        error_final = abs(mu_final - mu_true)
        print(f"Final Inferred Belief after 3 interactions: mu_hat = {mu_final:.3f}")
        print(f"Parameter Estimation Error: {error_initial:.3f} -> {error_final:.3f} ({(1 - error_final/error_initial)*100:.1f}% reduction)")

        # Now test 10-step multi-step rollout prediction:
        # Compare Unadapted (mu=0.50) vs Adapted (mu=mu_final) vs True Environment
        test_actions = [np.array([0.05, 0.05 * (i % 2 == 0)], dtype=np.float32) for i in range(10)]
        curr_true = state
        true_rollout = [curr_true]
        for a in test_actions:
            curr_true = sim.step(curr_true, a, mu_true)
            true_rollout.append(curr_true)
        true_rollout = np.array(true_rollout)

        # Unadapted rollout
        unadapted_rollout = [state]
        curr_unadapt = torch.tensor(state).unsqueeze(0)
        with torch.no_grad():
            for a in test_actions:
                t_a = torch.tensor(a).unsqueeze(0)
                curr_unadapt = model(curr_unadapt, t_a, torch.tensor([[0.50]]))
                unadapted_rollout.append(curr_unadapt.squeeze(0).numpy())
        unadapted_rollout = np.array(unadapted_rollout)

        # Adapted rollout
        adapted_rollout = [state]
        curr_adapt = torch.tensor(state).unsqueeze(0)
        with torch.no_grad():
            for a in test_actions:
                t_a = torch.tensor(a).unsqueeze(0)
                curr_adapt = model(curr_adapt, t_a, torch.tensor([[mu_final]]))
                adapted_rollout.append(curr_adapt.squeeze(0).numpy())
        adapted_rollout = np.array(adapted_rollout)

        # Compute MSE on shear force and displacement over rollout
        mse_unadapted = np.mean((unadapted_rollout[:, 1] - true_rollout[:, 1])**2)
        mse_adapted = np.mean((adapted_rollout[:, 1] - true_rollout[:, 1])**2)
        rollout_improvement = (1 - mse_adapted / mse_unadapted) * 100

        print(f"10-Step Shear Rollout MSE: Unadapted = {mse_unadapted:.4f} | Adapted = {mse_adapted:.4f}")
        print(f"Rollout Error Reduction: {rollout_improvement:.1f}%")

        results[tc["name"]] = {
            "mu_true": mu_true,
            "mu_final": mu_final,
            "error_final": error_final,
            "mse_unadapted": mse_unadapted,
            "mse_adapted": mse_adapted,
            "rollout_improvement": rollout_improvement,
            "mu_history": mu_history
        }

    return results

if __name__ == "__main__":
    sim = ContactPhysicsSimulator()
    model = ParameterConditionedDynamicsModel()
    
    # 1. Generate narrow training dataset
    s, a, m, ns = generate_training_data(sim, num_trajectories=400, traj_len=25, mu_range=(0.45, 0.55))
    
    # 2. Train parameter-conditioned model
    train_dynamics_model(model, s, a, m, ns, epochs=60)
    
    # 3. Run Experiment 0 evaluation
    results = run_experiment_0(sim, model)
    
    print("\n=======================================================")
    print("EXPERIMENT 0 FINAL SUMMARY")
    print("=======================================================")
    all_passed = True
    for name, res in results.items():
        passed = (res["error_final"] <= 0.08) and (res["rollout_improvement"] >= 40.0)
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        print(f"{name}: {status}")
        print(f"  Final mu Error: {res['error_final']:.3f} (Threshold: <= 0.08)")
        print(f"  Rollout MSE Reduction: {res['rollout_improvement']:.1f}% (Threshold: >= 40.0%)")

    print("\nOVERALL EXPERIMENT 0 OUTCOME:", "SIGNAL CONFIRMED (PROCEED)" if all_passed else "HYPOTHESIS FALSIFIED (KILL)")
