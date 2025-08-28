# Blade Parametric Modeling 🛠️

This folder contains blade models generated via the parameterized `../BladeGenerator.py` script. It allows flexible parametric modeling of blades for various designs.

---

## General Parameterization 🔧

Consider a single blade with angular span Θ and height along the z-axis H. The dimensionless radial parameter $r' \in [0,1]$ represents the normalized blade height (spanwise position).

For a single blade, we sample **5 cylindrical layers** along its height (see in the Figure). Each layer's shape curve is sampled in the θ-z plane as in Fig(a). Note that for a fixed blade, all layers share the same leading/trailing edge height. If the blade’s top and bottom are straight, the leading/trailing edge angle is the same. Twists can be added in the modeling process.

### Blade Geometry Definition 📐

- **Chord line:** Connects leading and trailing edge points.  
- **Leading edge coordinate:** $(θ_0(r'), z_0)$  
- **Camber and thickness:** Distance from the chord to the upper/lower surfaces. Denote half-thickness $t$ and vertical camber $h>0$.

The chord line equation:

```math
C(θ, r') = z_0 + \frac{(θ - θ_0)}{Θ} \cdot H
```

The upper/lower surface parameterization:

```math
z_{±}(θ, r') = -h(θ, r') ± t(θ, r')
```

Define maximum camber and thickness per layer:

```math
h(θ, r') = h_{\rm{max}}(r') \cdot γ_{r'}(θ), t(θ, r') = t_\rm{max}(r') * τ_{r'}(θ)
```

where $γ_{r'}(θ)$ and $τ_{r'}(θ)$ are relative camber and thickness functions:

```math
γ_{r'}(θ_0) = γ_{r'}(θ_0+Θ) = τ_{r'}(θ_0) = τ_{r'}(θ_0+Θ) = 0, γ_{r'}(θ), τ_{r'}(θ) ≤ 1
```

```math
\frac{d^2γ_{r'}}{dθ²}, \frac{d^2τ_{r'}}{dθ^2} ≤ 0, θ \in (θ_0, θ_0+Θ); \frac{dγ_r'}{dθ}, frac{dτ_{r'}}{dθ} \neq 0, θ ∈ {θ_0, θ_0+Θ}
```

So the final upper/lower surface becomes:

```math
z_{±}(r', θ) = z_0 + (θ - θ_0(r'))/Θ * H - h_{\rm{max}}(r') * γ_{r'}(θ) ± t_{\rm{max}(r') * τ_{r'}(θ)
```

---

## Relative Functions γ and τ 🎯

- Non-dimensional angle along the chord:

```math
θ'_r = \frac{(θ - θ_0(r'))}{Θ}
```
- Relative camber and thickness functions become:

```math
γ_{r'}(θ'_r) = γ_{r'}\left( \frac{(θ - θ_0(r'))}{Θ} \right), τ_{r'}(θ'_r) = τ_{r'}\left( \frac{(θ - θ_0(r'))}{Θ} \right)
```

- Boundary constraints:

```math
γ_{r'}(0) = γ_{r'}(1) = τ_{r'}(0) = τ_{r'}(1) = 0, sup γ_{r'} = sup τ_{r'} = 1
```

- Convexity:

```math
\frac{d^2γ_{r'}}{dθ'^2}, \frac{d^2τ_{r'}}{dθ'^2} ≤ 0, θ'_r ∈ (0,1); \frac{d^2γ_{r'}}{dθ'^2}, \frac{d^2τ_{r'}}{dθ'^2} < 0, θ'_r ∈ {0,1}
```

- Dimensionless surface equation:

```math
z_{±}(r', θ'_r) = θ_0(r') + θ'_r * H - h_\rm{max}(r') \cdot γ_{r'}(θ'_r) ± t_\rm{max}(r') \cdot τ_{r'}(θ'_r)
```

---

## Functional Forms ✨

- **Camber γ(r')** using Beta-like distribution (single parameter α):

```math
γ_{r'}(·) = \frac{(·)^{α_{r'}}(1 - ·)^{1 - α_{r'}}}{(α_{r'}^{α_{r'}}(1 - α_{r'})^{1 - α_{r'}})}
```

- **Thickness τ(r')** to keep uniform midspan:

```math
τ_{r'}(·) =
{
((1 - cos(π * · / a_{r'}))/2)^{β_{r'}}, (·) ∈ [0, a_{r'}]
1, (·) ∈ (a_{r'}, b_{r'})
((1 + cos(π * (· - b_{r'})/(1 - b_{r'})))/2)^{β_{r'}}, (·) ∈ [b_{r'}, 1]
}
```

- Parameters $a_{r'}, b_{r'}$ control the plateau, $β_{r'}$ controls steepness.

- Each layer has 7 parameters: $θ_0(r') h_\rm{max}(r') t_\rm{max}(r') α_{r'} a_{r'} b_{r'} β_{r'}$. With 5 layers plus global $Θ, H$, total **37 parameters per blade**.

---

## Improved Camber Function ⚡

- Original beta distribution sometimes causes over-bent tips.  
- Use **dual-parameter distribution**:

```math
γ_{r'}(·) = \frac{((·)^{κ·α}(1 - ·)^{κ·(1-α)})}{(α^α(1-α)^(1-α))^κ}
```

- Extra parameter $κ$ improves smoothness at blade tip. The parameters finally comes to **42**.

---

## Blade Surface Sampling 📐

- Sample **5 spans** along r'.  
- For each span, sample θ in θ-z plane.  
- Compute upper/lower surfaces using `γ_{r'}` and `τ_{r'}`.  
- Interpolate parameters linearly across layers.

---

## Figures and Placeholders 🖼️

- **Figure 2.4(a)**: Blade θ-z sampling (insert your image here)  
- **Figure 2.4(b)**: Blade chord and centerline illustration (insert your image here)

---

## Future Plans 🚀

- Develop a **Blade Modeling UI**  
- Compatibility with main software like **BladeGen**  
- Support **Bezier, spline-based designs**  

