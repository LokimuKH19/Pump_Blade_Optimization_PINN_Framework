# LFR Main Pump Ternary Impeller Structure Optimization 🚀💧

## 1. Background

### 1.1 Digital Twin Goals and Ternary Impeller Structure
Our ultimate goal is to build a digital twin system for the Lead-cooled Fast Reactor (LFR) main pump that integrates **simulation, design, and operational control** in one seamless workflow. This allows us to model the full life cycle of the pump and apply it to the analysis of the LFR primary loop system.  

So far, we've successfully used a **Multi-Input PINN (Physics-Informed Neural Network)** surrogate model to simulate and control the flow inside a given pump structure. By feeding in operating conditions like **flow rate and rotation speed** through extra input channels, the model can calculate velocity and pressure distributions in real-time, enabling multi-objective optimization of pump operation. ⚡

Now, we're taking it a step further. We want the surrogate model to **automatically predict flow fields for different structural parameters**. This enables faster calculations for fluid-structure interaction, corrosion analysis, and low-cost multi-objective structural optimization.

The challenge? Changing structural parameters messes with both the **solution domain and boundary conditions**, which is trickier than just tweaking operating conditions. To handle this, we need to:

- Identify key structural parameters 🔍  
- Handle **topology changes and domain variations**  
- Build datasets for complex boundary conditions  
- Adapt surrogate models for structure-driven optimization  

We classify structural parameters into two groups:

1. **Non-blade-related parameters**: affect installation rather than shape (e.g., number of blades, shaft radius, pump casing radius, blade tip clearance). These are sparse and mostly derived from industrial requirements.  
2. **Blade geometry parameters**: uniquely define blade shape (e.g., profile and thickness). These are dense and are the main source of complex boundary conditions.  

In short: non-blade parameters are easy, blade parameters are tricky. Our key strategy: **sparsify the blade description** to reduce optimization variables while maintaining accuracy, and decouple the two parameter groups wherever possible. 🎯

---

### 1.2 LFR Main Pump Structural Optimization
Unlike conventional centrifugal pumps, LFR pumps **care deeply about cavitation** and also face challenges from **high-temperature, high-density coolant flows** that cause fatigue and corrosion in complex geometries. Traditional pump design experience doesn’t transfer directly.  

Fatigue and corrosion are directly linked to flow patterns, meaning **flow optimization is unavoidable**. However, experimental data for lead and lead-bismuth alloys are limited, so simulating flow under various pump geometries is essential.  

To optimize, we need **precise, quantitative geometric parameterization**, enabling advanced optimization methods to work their magic. 🪄  

Our contribution?  

- **Reduce computation per simulation** 🏎️  
- **Shorten the overall design cycle** ⏱️  
- **Support complex, changing boundary conditions** 🌊  

In short: we make pump optimization faster, smarter, and slightly more magical. ✨