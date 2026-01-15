/*
  Copyright (C) 2025-2026 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/


#include <aspect/include/material_model/reaction_model/morgan2001_mantle_melting.h>
#include <aspect/utilities.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace ReactionModel
    {
      /*
      ==============================================================================
        Morgan2001 Multicomponent Mantle Melting Model
        ------------------------------------------
        This module implements a multi-component, pressure-, depletion-dependent,
        water-sensitive mantle melting parameterization adapted from:

          WMCM - 1D Decompression meltiing code for a wet multicomponent mantle
            Developed by J. Hasenclever & J.P. Morgan, 2009-2013, 2024-2025
            References: Morgan, J. P., 2001, G^3: "Thermodynamics of pressure
            release melting of a veined plum pudding mantle"
            Katz et al., 2003, G^3 (wet solidus)

        The melting updated at each time step solves for the melt increment dF
        at a single quadrature point using a Gauss–Newton iteration:
              e(F) = T(F) - Ts(F) = 0
        where
            T(F)  = updated temperature including latent heat of melting consumption
            Ts(F) = updated solidus including depletion & water effects.
      ==============================================================================
      */
      template <int dim>
      void
      Morgan2001MantleMelting<dim>::initialize ()
      {
        // Cache compositional field indices for cumulative melting degree (F),
        // solid bulk water content Xb, and if using incremental melting mode,
        // trapped melt fraction f_trapped, and Vol (volume fraction for each
        // solid residue).
        F_field_indices.resize(nc);
        f_trapped_field_indices.resize(nc);
        Xb_field_indices.resize(nc);
        Vol_field_indices.resize(nc);
        for (unsigned int i = 0; i < nc; ++i)
        {
          const std::string F_name = "F_component_" + std::to_string(i);
          const std::string f_name = "f_trapped_" + std::to_string(i);
          const std::string Xb_name = "Xb_" + std::to_string(i);
          const std::string Vol_name = "Vol_" + std::to_string(i);

          AssertThrow(this->introspection().compositional_name_exists(F_name),
                      ExcMessage("Compositional field <" + F_name + "> not found."));
          F_field_indices[i] =
            this->introspection().compositional_index_for_name(F_name);

          AssertThrow(this->introspection().compositional_name_exists(Xb_name),
                        ExcMessage("Compositional field <" + Xb_name + "> not found."));
          Xb_field_indices[i] =
            this->introspection().compositional_index_for_name(Xb_name); 

          if (melt_mode == "incremental")
          {
            AssertThrow(this->introspection().compositional_name_exists(f_name),
                        ExcMessage("Compositional field <" + f_name + "> not found."));
            f_trapped_field_indices[i] =
              this->introspection().compositional_index_for_name(f_name);

            AssertThrow(this->introspection().compositional_name_exists(Vol_name),
                        ExcMessage("Compositional field <" + Vol_name + "> not found."));
            Vol_field_indices[i] =
              this->introspection().compositional_index_for_name(Vol_name);
          }
        }
      }



      template <int dim>
      void
      Morgan2001MantleMelting<dim>::
      compute_current_solidus (const double              P_GPa_positive,
                               const std::vector<double> &Dpl,
                               const std::vector<double> &Xm,
                               std::vector<double>       &Ts_dry,
                               std::vector<double>       &Ts_wet,
                               std::vector<double>       &Ts_liquidus) const
      {
        Ts_dry.resize(nc);
        Ts_wet.resize(nc);
        Ts_liquidus.resize(nc);

        // water concentration in melt limited by pressure-dependent 
        // saturation (Xm_sat). Here we assume that melts from different
        // mantle components share the same Xm_sat value.
        const double Xm_sat = melt_water_saturation(P_GPa_positive);

        // Dpl = 1 for all lithologic components, used in liquidus calculation
        const std::vector<double> Dpl_liq(nc, 1.0);

        // Loop over each lithologic component
        for (unsigned int i=0; i<nc; ++i)
          {
            //---------------------Dry solidus-----------------------
            //    Ts_dry = Ts0 + dTs/dP * P + dTs/dDpl * Dpl
            // Ts0: reference solidus at P=0 [°C]
            // dTs/dP: solidus-pressure gradient [°C/GPa]
            // dTs/dDpl: solidus-depletion gradient [°C]
            //-------------------------------------------------------
            // Consider cpx-out and plag-to-spinel effects on dTs/dDpl
            double dTs_dDpl_modify = modify_dTs_dDpl_cpx_plg(i, P_GPa_positive, Dpl);
            Ts_dry[i] = Ts0[i] + dTs_dP_intrinsic[i] * P_GPa_positive
                               + dTs_dDpl_modify * Dpl[i];

            //---------------------Wet solidus-----------------------
            //    Ts_wet = Ts_dry + dT_Xm
            // dT_Xm: temperature lowering due to water in the melt
            // that is in chemical equilibrioum with the solid
            // (eq. 16 in Katz et al., 2003)
            //-------------------------------------------------------
            double Xm_cutoff = std::min(Xm[i], Xm_sat);
            Ts_wet[i] = Ts_dry[i] - 43.0 * std::pow(Xm_cutoff, 0.75);

            //---------------------Liquidus--------------------------
            //    Ts_liquidus = Ts0 + dTs/dP * P + dTs/dDpl * 1
            // if melting continued to Dpl = 1
            //-------------------------------------------------------
            double dTs_dDpl_1_modify = modify_dTs_dDpl_cpx_plg(i, P_GPa_positive, Dpl_liq);
            Ts_liquidus[i] = Ts0[i] + dTs_dP_intrinsic[i] * P_GPa_positive
                                    + dTs_dDpl_1_modify;
          }
      }



      template <int dim>
      double
      Morgan2001MantleMelting<dim>::
      melt_water_saturation (const double pressure_GPa) const
      {
        // Simplified Katz et al. 2003 parameterization, eq. 17
        // Pressure in GPa, output in wt.%
        return 12.0 * std::pow(pressure_GPa, 0.6) + 1.0 * pressure_GPa;
      }



      template <int dim>
      double
      Morgan2001MantleMelting<dim>::
      modify_dTs_dDpl_cpx_plg (const unsigned int i,
                               const double pressure_GPa,
                               const std::vector<double> &Dpl) const
      {
        // Modify the dTs/dDpl based on cpx-out and plg stability field effects
        // on one lithologic component 'i'
        // Start from the intrinsic solidus-depletion slope
        double dTs_dDpl_mod = dTs_dDpl_intrinsic[i];

        // --- Clinopyroxene-out productivity change ---
        // Ran out of cpx has reduced productivity (higher dTs/dDpl)
        if (include_cpx_out && Dpl[i] > Dpl_cpx_out[i])
          dTs_dDpl_mod *= a_cpx_out[i];
        
        // --- Plagioclase stability effect ---
        // At pressure below spinel to plagioclase (less fusible) phase
        //  transition, reduce productivity (higher dTs/dDpl)
        if (include_plag_field && pressure_GPa < P_spl2plag[i])
          dTs_dDpl_mod *= a_spl2plag[i];
        
        return dTs_dDpl_mod;
      }



      template <int dim>
      void Morgan2001MantleMelting<dim>::
      compute_batch_water_partitioning (const std::vector<double> &F,
                                        std::vector<double>       &Xs,
                                        std::vector<double>       &Xm,
                                        std::vector<double>       &dXs_dF,
                                        std::vector<double>       &dXm_dF) const
      {
        // ===================== BATCH MELTING ==============================
        // Physical assumptions:
        //  - Batch (equilibrium) melting in a closed system
        //  - Melt is NOT extracted; all melt remains with the solid residue
        //  - Solid and melt are always in chemical equilibrium
        //  - Bulk water content is conserved
        //  - Water is treated as a trace element with a bulk partition
        //    coefficient Dm_H2O
        //
        // In batch melting:
        //  - F is the cumulative degree of melting
        //  - F is also equal to f_trapped (the trapped melt fraction in the solid)
        // ==================================================================

        // Resize output vectors
        Xs.resize(nc); // water concentration in the solid
        Xm.resize(nc); // water concentration in the melt
        dXs_dF.resize(nc);
        dXm_dF.resize(nc);

        for (unsigned int i = 0; i < nc; ++i)
          {
            // Governing equations:
            // (1) Equilibrium partitioning:
            //     Xs = D_H2O * Xm
            // (2) Mass conservation (closed system):
            //     Xb0 = (1 - F) * Xs + F * Xm
            //
            // Solving for Xm and Xs gives:
            //     Xm(F) = Xb0 / [ D_H2O + F * (1 - D_H2O) ]
            //     Xs(F) = D_H2O * Xm(F)
            //           = Xb0 * D_H2O / [ D_H2O + F * (1 - D_H2O) ]
            //
            // where Xb0 is the initial bulk water content.
            // See eq. 18 in Katz et al., 2003.
            // -----------------------------------------------------------------
            Xs[i] = Xb0[i] * Dm_H2O[i] / (Dm_H2O[i] + F[i] * (1.0 - Dm_H2O[i]));
            Xm[i] = Xs[i] / Dm_H2O[i];

            // These derivatives describe how water concentrations change
            // as melting progresses under batch-melting conditions.
            dXs_dF[i] = (Dm_H2O[i] * Xb0[i] * (Dm_H2O[i] - 1.0)) 
                        / std::pow((Dm_H2O[i] + F[i] * (1.0 - Dm_H2O[i])), 2);
            dXm_dF[i] = dXs_dF[i] / Dm_H2O[i];
          }
      }



      template <int dim>
      void Morgan2001MantleMelting<dim>::
      compute_fractional_water_partitioning (const std::vector<double> &F,
                                             std::vector<double>       &Xs,
                                             std::vector<double>       &Xm,
                                             std::vector<double>       &dXs_dF,
                                             std::vector<double>       &dXm_dF) const
      {
        // ===================== FRACTIONAL MELTING ========================
        // Physical assumptions:
        //  - Fractional melting in an open system
        //  - Melt is instantly extracted; only the infinitesimal new melt
        //    increment instantaneously equilibrates with the solid residue
        //  - Water behaves as a trace element with partition coefficient Dm_H2O
        //
        // In fractional melting:
        //  - F is the cumulative degree of melting
        //  - Xs is the residual solid water concentration after continuous extraction
        // ==================================================================

        // Resize all output vectors
        Xs.resize(nc); // water concentration in the solid
        Xm.resize(nc); // water concentration in the instantaneous melt
        dXs_dF.resize(nc);
        dXm_dF.resize(nc);

        for (unsigned int i = 0; i < nc; ++i)
          {
            // Governing equations:
            // (1) remaining water in the residual solid (Shaw, 1970):
            //     Xs(F) = Xb0 * (1 - F)^(1/Dm_H2O - 1)
            // where Xb0 is the initial bulk water content and equals
            // initial solid water Xs0.
            // (2) Instantaneous equilibrium:
            //     Xm = Xs / Dm_H2O
            Xs[i] = Xb0[i] * std::pow(1.0 - F[i], 1.0 / Dm_H2O[i] - 1.0);
            Xm[i] = Xs[i] / Dm_H2O[i];

            // These derivatives describe how water concentrations change
            // as melting progresses under fractional-melting conditions.
            dXs_dF[i] = - Xb0[i] * (1.0 / Dm_H2O[i] - 1.0)
                        * std::pow(1.0 - F[i], 1.0 / Dm_H2O[i] - 2.0);
            dXm_dF[i] = dXs_dF[i] / Dm_H2O[i];
          }
      }



      template <int dim>
      void Morgan2001MantleMelting<dim>::
      compute_incremental_water_partitioning (const std::vector<double> &f_trapped,
                                              const std::vector<double> &Xb,
                                              std::vector<double>       &Xs,
                                              std::vector<double>       &Xm,
                                              std::vector<double>       &dXs_dF,
                                              std::vector<double>       &dXm_dF) const
      {
        // ===================== INCREMENTAL MELTING ========================
        // A fraction of the melt is retained in the solid while the rest
        // escapes. The trapped melt with volume fraction f_trapped remains 
        // in equilibrium with the rock with volume fraction 1 - f_trapped.
        // 
        // Two regimes are distinguished:
        // (1) f_trapped < f_trapped_max
        //     - f_trapped grows with melting degree F -> df_trapped/dF = 1
        //     - Solid and trapped melt remain in equilibrium:
        //         Xm = Xs / D
        //         Xb = (1 - f_trapped) * Xs + f_trapped * Xm
        //     - Chain rule applies:
        //         dXm/dF = dXm/df_trapped
        //
        // (2) f_trapped ≥ f_trapped_max
        //     - f_trapped is fixed, df_trapped/dF = 0
        //     - Newly produced melt is immediately extracted and does not
        //       participate in equilibrium
        //     - Trapped melt composition is frozen:
        //         dXs/dF = dXm/dF = 0
        //
        // Input:
        //        f_trapped: volume fraction of trapped melt in the solid
        //        Xb : bulk water content in each component (wt.%)
        // Output:
        //        Xs : water concentration in solid (wt.%)
        //        Xm : water concentration in melt (wt.%)
        //        dXs_dF, dXm_dF : derivatives with respect to F
        // ==================================================================
        // Resize all output vectors
        Xs.resize(nc);
        Xm.resize(nc);
        dXs_dF.resize(nc);
        dXm_dF.resize(nc);
        for (unsigned int i = 0; i < nc; ++i)
        {
          const double D = Dm_H2O[i];
          const double f = f_trapped[i];
          // Common equilibrium denominator:
          const double denom = 1.0 - f + f / D;

          Xs[i] = Xb[i] / denom;
          Xm[i] = Xs[i] / D;

          if (f < f_trapped_max)
          {
            // dXs_dF = dXs_df_trapped : change of Xs w.r.t. f_trapped
            // dXm_dF = dXm_df_trapped : change of Xs w.r.t. f_trapped
            dXs_dF[i] = - Xb[i] * (1.0 / D - 1.0) / (denom * denom);
            dXm_dF[i] = dXs_dF[i] / D;
          }
          else
          {
            // Trapped melt volume fraction is fixed to f_trapped_max
            dXs_dF[i] = 0.0;
            dXm_dF[i] = 0.0;
          }
        }
      }



      template <int dim>
      std::vector<double>
      Morgan2001MantleMelting<dim>::
      equilibrate_water_between_solids (const std::vector<double> &Vol,
                                        const std::vector<double> &f_trapped,
                                        const std::vector<double> &Xs) const
      {
        // ---------------------------------------------------------------------
        // Diffusive re-equilibration of water between solid lithologic components
        //
        // This function redistributes water stored in the solid fraction among
        // coexisting lithologic components at a single node (or quadrature point).
        // The redistribution represents subgrid-scale diffusive equilibration
        // of water between solids over a finite timescale.
        //
        // Key assumptions:
        //  - Only the solid fraction (1 - f_trapped) participates in re-equilibration.
        //  - The total amount of water stored in solids is strictly conserved.
        //  - A fully equilibrated reference state is defined by prescribed
        //    solid–solid partition coefficients Ds_H2O.
        //  - Re-equilibration during a single timestep is partial and controlled
        //    by the dimensionless equilibration fraction f_equil (0 ≤ f_equil ≤ 1),
        //    which parameterizes the ratio of the timestep to the diffusive
        //    equilibration timescale.
        //
        // Numerical implementation:
        //  - The current solid water concentrations are relaxed linearly towards
        //    the fully equilibrated state.
        //  - A small iterative correction is applied to eliminate numerical
        //    deviations from mass conservation introduced by the relaxation step.
        // ---------------------------------------------------------------------

        // Single component: no solid–solid equilibration possible
        if (nc == 1)
          return Xs;

        // Calculate the equilibration fraction f_equil (0 ≤ f_equil ≤ 1).
        // The equilibration timescale t_equil corresponds to diffusive 
        // equilibration over a characteristic length scale L0_equil with a
        // constant water diffusivity of 1e-7 m^2/s.
        const double dt = this->get_timestep();  // current timestep size [s]
        const double t_equil = (L0_equil * L0_equil) / 1e-7; //[s]
        const double f_equil = std::min(1.0, std::max(0.0, 1.0 - std::exp(- dt / t_equil)));

        // Only the solid fraction participates in re-equilibration
        std::vector<double> Vol_solids(nc);
        for (unsigned int i = 0; i < nc; ++i)
          Vol_solids[i] = Vol[i] * (1.0 - f_trapped[i]);

        // Total water stored in solids (must be conserved)
        double Xs_total = 0.0;
        for (unsigned int i = 0; i < nc; ++i)
          Xs_total += Vol_solids[i] * Xs[i];

        // Fully equilibrated reference state defined by Ds_H2O
        // Enforce:
        //   Xs_equil_i / Xs_equil_0 = Ds_H2O_i / Ds_H2O_0
        // and conservation of total solid water
        double denominator = Vol[0];
        for (unsigned int i = 1; i < nc; ++i)
          denominator += Vol[i] * (Ds_H2O[i] / Ds_H2O[0]);

        AssertThrow(denominator > 0.0,
                    ExcMessage("Invalid denominator in solid water equilibration"));

        std::vector<double> Xs_equil(nc);
        Xs_equil[0] = Xs_total / denominator;
        for (unsigned int i = 1; i < nc; ++i)
          Xs_equil[i] = Xs_equil[0] * (Ds_H2O[i] / Ds_H2O[0]);

        // Partial re-equilibration (diffusive relaxation)
        std::vector<double> Xs_out(nc);
        for (unsigned int i = 0; i < nc; ++i)
          Xs_out[i] = (1.0 - f_equil) * Xs[i] + f_equil * Xs_equil[i];

        // Iterative correction to enforce exact mass conservation
        const double tol_Xs = 1e-8;
        const double wght   = 0.9;
        for (unsigned int it = 0; it < 20; ++it)
        {
          double Xs_err = 0.0;
          for (unsigned int i = 0; i < nc; ++i)
            Xs_err += Vol_solids[i] * Xs_out[i];
          Xs_err -= Xs_total;

          // Converged: total solid water conserved
          if (std::abs(Xs_err) < tol_Xs)
            return Xs_out;

          // Linear correction along the Ds_H2O direction
          for (unsigned int i = 0; i < nc; ++i)
            Xs_out[i] += -wght * Xs_err * (Ds_H2O[i] / Ds_H2O[0]);
        }

        // Final safety check
        double final_err = 0.0;
        for (unsigned int i = 0; i < nc; ++i)
          final_err += Vol_solids[i] * Xs_out[i];
        final_err -= Xs_total;

        AssertThrow(std::abs(final_err) < tol_Xs,
                    ExcMessage("Solid water equilibration failed to conserve mass"));

        return Xs_out;
      }



      template <int dim>
      std::vector<double>
      Morgan2001MantleMelting<dim>::
      calculate_dF_iterative (const double T_Celsius,
                              const double P_GPa_positive,
                              const double Cp,
                              const std::vector<double> &F,
                              const std::vector<double> &f_trapped,
                              const std::vector<double> &Vol,
                              const std::vector<double> &Xm,
                              const std::vector<double> &Xm_sat,
                              const std::vector<double> &Xb,
                              std::vector<double>       &Xm_equil,
                              std::vector<double>       &dXs_dF_equil) const
      {
        // Changes in cumulative degree of melting F
        std::vector<double> dF(nc, 0.0);

        // equilibrium outputs used for updating Xb
        Xm_equil = Xm;                // for incremental mode
        dXs_dF_equil.assign(nc, 0.0); // for fractional mode

        // Current cumulative degree of depletion (used for solidus)
        // Dpl = Dpl0 + F, limited to [0,1]
        std::vector<double> F_cutoff(nc), Dpl(nc);
        // F is advected from the previous timestep and may violate physical 
        // bounds due to numerical diffusion or interpolation. We therefore
        // project F onto the physically admissible range before entering the 
        // Newton iteration.
        for (unsigned int i = 0; i < nc; ++i)
        {
          F_cutoff[i] = std::min(1.0 - Dpl0[i], std::max(0.0, F[i]));
          Dpl[i] = Dpl0[i] + F_cutoff[i];
        }

        // Compute current dry / wet solidus and liquidus temperatures
        std::vector<double> Ts_dry(nc), Ts_wet(nc), Ts_liquidus(nc);
        compute_current_solidus(P_GPa_positive, Dpl, Xm,
                                Ts_dry, Ts_wet, Ts_liquidus);

        //---------------------------------------------------------------------
        // Step 1. Determine which components are eligible for melting
        //---------------------------------------------------------------------
        std::vector<unsigned int> melt_index;
        // Volume fraction below which a mantle component is treated as "vanished"
        const double Vol_cutoff = 1e-3;  // 0.1% vol.%

        for (unsigned int i = 0; i < nc; ++i)
          if (T_Celsius > Ts_wet[i] && Dpl[i] < 1.0 && Vol[i] > Vol_cutoff)
            melt_index.push_back(i);

        // number of melting components at this point
        const unsigned int n_melt = melt_index.size();

        //---------------------------------------------------------------------
        // Step 2. If no components can melt --> return zero dF vector
        //---------------------------------------------------------------------
        if (n_melt == 0)
          return dF;  // no melting at this point

        //---------------------------------------------------------------------
        // Step 3. Newton iteration to solve for dF of multi-component melting
        //---------------------------------------------------------------------
        // Solve for dF_i by enforcing local thermodynamic equilibrium:
        //
        //   e_i(dF) = T(dF) - Ts_i(dF) <= T_tolerance
        //
        // T(dF) includes latent heat of melting; Ts_i depends on pressure,
        // depletion, and melt water.
        // The Jacobian J_ij = ∂e_i/∂dF_j has two parts:
        //
        // (1) Latent-heat coupling (all components):
        //     ∂T/∂dF_j = - ΔH_j * Vol_j / Cp
        //             = - (T_K * ΔS_j * Vol_j) / Cp
        //
        // (2) Solidus term (diagonal only):
        //     ∂Ts_i/∂dF_j = δ_ij [dTs_i/dDpl_i + dTs_i/dXm_i * dXm_i/dF_i]
        // since Dpl_i = Dpl0_i + F_i.
        //
        // The depletion term raises Ts (cpx-out, phase effects), and the
        // water term lowers Ts; dXm_i/dF_i depends on melting mode.
        // Non-melting components are excluded from residuals and Jacobian.

        // ∂T/∂F due to latent heat of melting consumption
        std::vector<double> dT_dF(nc);
        for (unsigned int i = 0; i < nc; ++i)
          dT_dF[i] = - dH[i] * Vol[i] / Cp;

        // Newton iteration variables (melting components only)
        std::vector<double> dF_it(n_melt, 1e-3);   // initial guess
        std::vector<unsigned int> bound_hits(n_melt, 0);
        Vector<double> residual(n_melt); // residual vector e
        Vector<double> newton_F(n_melt); // Newton F increment
        FullMatrix<double> J(n_melt, n_melt); // Jacobian matrix

        // Trial state (full nc, but only part updated in the Newton loop)
        std::vector<double> F_it(nc), Dpl_it(nc), f_trapped_it(nc);
        std::vector<double> Xs_it(nc), Xm_it(nc);
        std::vector<double> dXs_dF_it(nc, 0.0), dXm_dF_it(nc, 0.0);

        // Newton loop starts
        const unsigned int max_iter = 20; // maximum iterations
        const double tol_T = 1e-6; // temperature tolerance [C]
        for (unsigned int it = 0; it < max_iter; ++it)
        {
          // --- Initialize trial state ---
          // Each Newton iteration must start from the same base state
          // (previous converged timestep), NOT from the previous iteration.
          // Otherwise, Newton linearization becomes inconsistent and may diverge.
          for (unsigned int i = 0; i < nc; ++i)
          {
            F_it[i]   = F_cutoff[i];
            Dpl_it[i] = Dpl[i];
            // Clamp base-state trapped melt fraction to [0, f_trapped_max].
            if (melt_mode == "incremental")
              f_trapped_it[i] = std::min(std::max(0.0, f_trapped[i]), f_trapped_max);
            else if (melt_mode == "batch")
              f_trapped_it[i] = F_it[i];
            else if (melt_mode == "fractional")
              f_trapped_it[i] = 0.0;
          }

          // --- Apply trial dF only to melting components ---
          for (unsigned int a = 0; a < n_melt; ++a)
          {
            const unsigned int k = melt_index[a];
            // enforce bounds: no freezing, no over-melting
            dF_it[a] = std::min(std::max(0.0, dF_it[a]), 1.0 - F_cutoff[k]);

            F_it[k]   = F_cutoff[k] + dF_it[a];
            Dpl_it[k] = std::min(1.0, Dpl0[k] + F_it[k]);
            if (melt_mode == "incremental")
              f_trapped_it[k] = std::min(f_trapped_max, f_trapped[k] + dF_it[a]);
            else if (melt_mode == "batch")
              f_trapped_it[k] = F_it[k];
            else if (melt_mode == "fractional")
              f_trapped_it[k] = 0.0;
          }

          // --- Recompute water partitioning with trial state ---
          if (melt_mode == "batch")
            compute_batch_water_partitioning(F_it, Xs_it, Xm_it, dXs_dF_it, dXm_dF_it);
          else if (melt_mode == "fractional")
            compute_fractional_water_partitioning(F_it, Xs_it, Xm_it, dXs_dF_it, dXm_dF_it);
          else if (melt_mode == "incremental")
            compute_incremental_water_partitioning(f_trapped_it, Xb, Xs_it, Xm_it, dXs_dF_it, dXm_dF_it);

          // --- Apply water saturation cutoff ---
          for (unsigned int i = 0; i < nc; ++i)
            if (Xm_it[i] > Xm_sat[i])
            {
              Xm_it[i] = Xm_sat[i];
              dXm_dF_it[i] = 0.0;
            }

          // --- Recompute solidus with trial state ---
          compute_current_solidus(P_GPa_positive, Dpl_it, Xm_it,
                                  Ts_dry, Ts_wet, Ts_liquidus);

          // --- Update semi-implicit temperature ---
          // Including updated latent heat of melting consumption
          double T_it = T_Celsius;
          for (unsigned int a = 0; a < n_melt; ++a)
            T_it += dT_dF[melt_index[a]] * dF_it[a];

          // --- residual: T_it - Ts ---
          for (unsigned int a = 0; a < n_melt; ++a)
            residual[a] = T_it - Ts_wet[melt_index[a]];

          // --- Check if convergence is achieved ---
          bool converged = true;
          for (unsigned int a = 0; a < n_melt; ++a)
            if (std::abs(residual[a]) > tol_T)
              converged = false;

          if (converged)
            break;

          // --- Compute solidus derivatives at trial state ---
          std::vector<double> dTs_dF_it(nc, 0.0);
          for (unsigned int i = 0; i < nc; ++i)
          {
            // dTs_dF = dTs_dDpl, since Dpl = Dpl0 + F
            // dTs/dDpl: solidus-depletion gradient [°C]
            // Consider cpx-out and plag-to-spinel effects on dTs/dDpl
            dTs_dF_it[i] = modify_dTs_dDpl_cpx_plg(i, P_GPa_positive, Dpl_it);
            
            // Solidus shift due to water in melt (Katz et al., 2003, eq. 16)
            // Use cutoff to avoid singular derivative as Xm -> 0
            const double Xm_cutoff = std::max(Xm_it[i], 1e-4); // wt.%
            const double dTs_dXm = - 43.0 * 0.75 
                                   * std::pow(Xm_cutoff, -0.25); // °C / wt.%

            dTs_dF_it[i] += dTs_dXm * dXm_dF_it[i];
          }
          
          // --- Jacobian: J_ab = ∂Teff / ∂dF_b - ∂Ts_a / ∂F_b ---
          J = 0.0;
          for (unsigned int a = 0; a < n_melt; ++a)
          {
            const unsigned int m = melt_index[a];
            for (unsigned int b = 0; b < n_melt; ++b)
            {
              const unsigned int n = melt_index[b];
              J(a,b) = dT_dF[n]; // latent heat term (all entries)
              if (a == b)
                J(a,b) -= dTs_dF_it[m]; // solidus derivative (diagonal only)
            }
          }
  
          // --- Newton update: dF ← dF − J⁻¹ e---
          J.gauss_jordan(); // small matrices only (e.g., n_melt ≤ 3)
          J.vmult(newton_F, residual); // newton_F = J^{-1} * residual
          for (unsigned int a = 0; a < n_melt; ++a)
            dF_it[a] -= newton_F[a];

          // --- enforce physical bounds on dF_it ---
          bool any_melting_left = false;
          for (unsigned int a = 0; a < n_melt; ++a)
          {
            const unsigned int k = melt_index[a];
            const double dF_max = 1.0 - F_cutoff[k];

            // project onto admissible interval
            const double dF_old = dF_it[a];
            dF_it[a] = std::min(std::max(0.0, dF_it[a]), dF_max);

            // disable component if Newton repeatedly violates bounds
            if (dF_it[a] != dF_old)
              bound_hits[a] += 1;
            else
              bound_hits[a] = 0;

            if (bound_hits[a] >= 3)
              component_is_melting[k] = false;

            if (component_is_melting[k])
              any_melting_left = true;
          }

          if (!any_melting_left)
            break;
        }

        // After Newton convergence, store equilibrium outputs
        Xm_equil = Xm_it;
        dXs_dF_equil = dXs_dF_it;
        
        //---------------------------------------------------------------------
        // Step 4. Scatter back to full dF vector
        //---------------------------------------------------------------------
        for (unsigned int j = 0; j < n_melt; ++j)
          dF[melt_index[j]] = dF_it[j];

        return dF;
      }



      template <int dim>
      void
      Morgan2001MantleMelting<dim>::
      fill_reaction_outputs (const typename Interface<dim>::MaterialModelInputs &in,
                             typename Interface<dim>::MaterialModelOutputs &out) const
      {
        for (unsigned int q=0; q<in.n_evaluation_points(); ++q)
          {
            //-----------------------------------------------------------------
            // 0. Preprocessing: unit conversions & initializations
            //-----------------------------------------------------------------
            // Pressure in GPa must be positive
            const double P_GPa_positive = std::max(0.0, in.pressure[q] / 1e9);

            // Temperature in Celsius.
            const double T_Celsius = in.temperature[q] - 273.15; // K → °C
            const double Cp = out.specific_heat[q]; // J/(kg·K)

            // Read specific compositional fields:
            // 1. Current cumulative degree of melting - F
            // 2. Solid bulk water content - Xb
            // If using incremental melting mode:
            // 3. Volume fraction of trapped melt in solid - f_trapped
            // 4. Volume fraction for each solid residue - Vol
            std::vector<double> F(nc), Xb(nc), f_trapped(nc), Vol(nc);
            for (unsigned int i = 0; i < nc; ++i)
            {
              Vol[i]       = Vol0[i];
              F[i]         = in.composition[q][F_field_indices[i]];
              if (melt_mode == "incremental")
              {
                f_trapped[i] = in.composition[q][f_trapped_field_indices[i]];
                Xb[i]        = in.composition[q][Xb_field_indices[i]];
                Vol[i]       = in.composition[q][Vol_field_indices[i]];
              }
              else if (melt_mode == "batch")
              {
                f_trapped[i] = F[i];
                Xb[i]        = Xb0[i];
              }
              else if (melt_mode == "fractional")
              {
                f_trapped[i] = 0.0;
                Xb[i]        = in.composition[q][Xb_field_indices[i]];
              }
            }

            //-----------------------------------------------------------------
            // 1. Water partitioning at current state (for solidus etc.)
            //-----------------------------------------------------------------
            std::vector<double> Xs(nc), Xm(nc), dXs_dF(nc), dXm_dF(nc);

            // Water content in melt at saturation (upper limit for Xm)
            // Here we assume that melts from different mantle components
            // share the same Xm_sat value.
            const double Xm_sat_val = melt_water_saturation(P_GPa_positive);
            std::vector<double> Xm_sat(nc, Xm_sat_val);

            // water concentrations will be advected with F and f_trapped
            if (melt_mode == "batch")
              compute_batch_water_partitioning(F, Xs, Xm, dXs_dF, dXm_dF);
            else if (melt_mode == "fractional")
              compute_fractional_water_partitioning(F, Xs, Xm, dXs_dF, dXm_dF);
            else if (melt_mode == "incremental")
              compute_incremental_water_partitioning(f_trapped, Xb, Xs, Xm,
                                                     dXs_dF, dXm_dF);
            else
              AssertThrow(false, ExcMessage("Melting mode not recognized."));
 
            // Equilibrate Xs between different components if needed
            if (equilibrate_Xs)
            {
              Xs = equilibrate_water_between_solids(Vol,f_trapped,Xs);
              // Maintain an instantaneous local solid–liquid equilibrium
              // after Xs changes
              for (unsigned int i = 0; i < nc; ++i)
                Xm[i] = Xs[i] / Dm_H2O[i];
            }

            //-----------------------------------------------------------------
            // 2. solve for changes in degree of melting (dF) at this point
            //-----------------------------------------------------------------
            std::vector<double> dF(nc, 0.0), Xm_equil(nc, 0.0), dXs_dF_equil(nc, 0.0);
            dF = calculate_dF_iterative(T_Celsius, P_GPa_positive, Cp, F, f_trapped,
                                          Vol, Xm, Xm_sat, Xb, Xm_equil, dXs_dF_equil);

            //-----------------------------------------------------------------
            // 3. Write reaction terms to specific compositional fields
            //-----------------------------------------------------------------
            std::vector<double> dF_limited(nc, 0.0), dF_extracted(nc, 0.0);
            for (unsigned int i = 0; i < nc; ++i)
              dF_limited[i] = std::min(std::max(0.0, dF[i]), 1.0 - F[i]);

            // Compute extracted melt fraction beyond trapped capacity
            if (melt_mode == "incremental")
              for (unsigned int i = 0; i < nc; ++i)
                dF_extracted[i] = std::max(0.0, std::min(dF_limited[i],
                                                         f_trapped[i]
                                                         + dF_limited[i]
                                                         - f_trapped_max));

            // Write reaction terms for each component
            for (unsigned int i = 0; i < nc; ++i)
            {
              // Cumulative degree of melting evolution
              const unsigned int F_index = F_field_indices[i];
              out.reaction_terms[q][F_index] = dF_limited[i];

              // Bulk water content evolution
              // Note that in fractional mode, Xb = residue solid water Xs
              // Incremental mode: Xb decreases by extracted melt beyond trapped capacity
              const unsigned int Xb_idx = Xb_field_indices[i];
              if (melt_mode == "batch")
                out.reaction_terms[q][Xb_idx] = 0.0;
              else if (melt_mode == "fractional")
                out.reaction_terms[q][Xb_idx] = dXs_dF_equil[i] * dF_limited[i];
              else if (melt_mode == "incremental")
                out.reaction_terms[q][Xb_idx] = - dF_extracted[i] * Xm_equil[i];

              // Trapped melt fraction evolution (incremental melting only)
              if (melt_mode == "incremental")
              {
                const unsigned int f_index = f_trapped_field_indices[i];
                if (f_trapped[i] < f_trapped_max)
                  out.reaction_terms[q][f_index] = dF_limited[i];
                else
                  out.reaction_terms[q][f_index] = 0.0;
              }
            }

            // Solid volume fraction evolution (incremental melting only)
            if (melt_mode == "incremental")
            {
              double Vol_sum = 0.0;
              // Compute total residual volume after extraction
              for (unsigned int i = 0; i < nc; ++i)
                Vol_sum += Vol[i] * (1.0 - dF_extracted[i]);

              const double inv_Vol_sum = (Vol_sum > 0.0 ? 1.0 / Vol_sum : 0.0);

              // Renormalize component volumes and write dVol.
              for (unsigned int i = 0; i < nc; ++i)
              {
                const unsigned int Vol_index = Vol_field_indices[i];
                const double Vol_new = Vol[i] * (1.0 - dF_extracted[i]) * inv_Vol_sum;
                out.reaction_terms[q][Vol_index] = Vol_new - Vol[i];
              }
            }
          }
      }



      template <int dim>
      void Morgan2001MantleMelting<dim>::
      declare_parameters(ParameterHandler &prm)
      {
        prm.enter_subsection("Mantle melting settings");
        {
          prm.declare_entry ("Melting mode", "fractional",
                            Patterns::Selection("batch|fractional|incremental"),
                            "Controls how water and incompatible elements partition "
                            "between solid and melt during melting. Supported modes are:\n"
                            "  batch: Melt is retained in the residue; all melt remains "
                            "in chemical equilibrium with the solid (closed system melting).\n"
                            "  fractional: Melt is instantly extracted; only the infinitesimal "
                            "new melt increment equilibrates with the solid (open system melting).\n"
                            "  incremental: A fraction of melt is retained while the rest escapes; "
                            "water equilibrates only with the trapped melt. Requires specification "
                            "of trapped melt volume fraction f_trapped.\n"
                            "The mode determines how Xs (solid water), Xm (melt water), "
                            "and dXm/dF are computed in the melt solver.\n");
          // ----------------------------------------------------------
          // --- Fundamental petrological settings --------------------
          // ----------------------------------------------------------
          prm.declare_entry ("Number of lithologic components", "1",
                            Patterns::Double (),
                            "Number of lithologic (petrological) components (nc) in the mantle. "
                            "Each component has its own solidus, depletion law, and water "
                            "partitioning parameters.  Typical values:\n"
                            "  nc = 1 : homogeneous peridotite (single nonlinear equation).\n"
                            "  nc = 2 : peridotite + pyroxenite (2x2 Gauss-Newton system).\n"
                            "  nc = 3 : multi-component mantle following Morgan (2001).\n"
                            "Higher values are supported if all petrological parameters are "
                            "provided with dimension 'nc'.  Increasing nc increases the cost "
                            "of the melt solver. Currently, we support nc = 1, 2, or 3.\n"
                            "Units: None.");
          // All petrological parameter lists below must match the length of nc.
          prm.declare_entry ("Initial volume fraction Vol0", "1.0",
                            Patterns::List(Patterns::Double()),
                            "Volume fractions of each component. "
                            "Length = nc. Units: None.");
          prm.declare_entry ("Initial depletion Dpl0", "0.0",
                            Patterns::List(Patterns::Double()),
                            "Initial mantle depletion/melt extraction for each "
                            "component. Length = nc. Units: None.");
          prm.declare_entry ("Initial melt water Xm0", "0.0",
                            Patterns::List(Patterns::Double()),
                            "Initial water content of the melt for each component. "
                            "Length = nc. $\\text{wt\\%}$.");
          prm.declare_entry ("Initial bulk water Xb0", "0.0",
                            Patterns::List(Patterns::Double()),
                            "Initial bulkwater content for each component. "
                            "Length = nc. $\\text{wt\\%}$.");
          prm.declare_entry ("Water partition coefficient Dm_H2O", "0.01",
                            Patterns::List(Patterns::Double()),
                            "Bulk H2O partition coefficient between solid and melt for "
                            "each component: D = Xs / Xm. Length = nc. Units: None.");
          prm.declare_entry ("Water partition coefficient Ds_H2O", "1.0",
                            Patterns::List(Patterns::Double()),
                            "Bulk H2O partition coefficient between solids for each "
                            "component. Here we use the first component as reference, "
                            "so Ds(0) = 1.0. Ds(1) = Xs(1) / Xs(0). If there exists,  "
                            "Ds(2) = Xs(2) / Xs(0). Length = nc. Units: None.");
          prm.declare_entry ("Equilibrate solid water between components", "true",
                            Patterns::Bool (),
                            "Specify whether the lithologic component participates in water"
                            "re-equilibration between solids.");
          prm.declare_entry ("Water diffusive length scale L0_equil", "1000.0",
                            Patterns::Double(),
                            "Characteristic equilibration length scale for solid water. "
                            "This parameter represents the effective diffusive or "
                            "reactive length scale controlling water exchange between "
                            "solids, i.e., an average veins thickness or blob size of "
                            "the different lithologies. To achieve a sudden complete "
                            "re-equilibration use a small value (~1). Units: meters (m).");
          prm.declare_entry ("Maximum trapped melt volume fraction", "0.05",
                            Patterns::List(Patterns::Double()),
                            "Maximum volume fraction of the trapped melt remaining in equilibrium "
                            "with residual rock (all additional melt is extraced). This parameter "
                            "is used in the melting mode of "incremental". Units: None");
          prm.declare_entry("Latent heat of fusion", "660000",
                            Patterns::List(Patterns::Double()),
                            "Latent heat for each mantle component. "
                            "Length = nc. Units: $\\frac{\\text{J}}{\\text{kg}}$.");

          // ----------------------------------------------------------
          // --- Solidus & liquidus parameters ------------------------
          // ----------------------------------------------------------
          prm.declare_entry("Solidus temperature at surface", "1081.0",
                            Patterns::List(Patterns::Double()),
                            "Surface solidus for each component. Length = nc. "
                            "Units: $^{\\circ}C$.");
          prm.declare_entry("Solidus-pressure gradients", "132.0",
                            Patterns::List(Patterns::Double()),
                            "Solidus-pressure gradient for each component dTs/dP. "
                            "Length = nc. Units: $\\frac{^{\\circ}C}{\\text{GPa}}$.");
          prm.declare_entry("Solidus-depletion gradients", "250.0",
                            Patterns::List(Patterns::Double()),
                            "Solidus-depletion gradient for each component dTs/dDpl. "
                            "Length = nc. Units: $^{\\circ}C$.");

          // ----------------------------------------------------------
          // --- Phase boundary corrections ---------------------------
          // ----------------------------------------------------------
          prm.declare_entry ("Include cpx-out effect", "false",
                            Patterns::Bool (),
                            "Specify whether to include clinopyroxene exhaustion effect.");
          prm.declare_entry ("Include plag-to-spinel effect", "false",
                            Patterns::Bool (),
                            "Specify whether to include plagioclase stability field effect.");
          prm.declare_entry("Dpl_cpx_out", "0.18",
                            Patterns::List(Patterns::Double()),
                            "Degree of depletion where clinopyroxene is exhausted. "
                            "Beyond this degree, dTs/dDpl is increased by factor a_cpx_out. "
                            "Length = nc. Units: None.");
          prm.declare_entry("a_cpx_out", "5.0",
                            Patterns::List(Patterns::Double()),
                            "Multiplier factor increasing dTs/dDpl after CPX-out. "
                            "Length = nc. Units: None.");
          prm.declare_entry("P_spl2plag", "0.65",
                            Patterns::List(Patterns::Double()),
                            "Pressure boundary between spinel and plagioclase stability field. "
                            "See Borhini et al. (2010), doi:10.1093/petrology/egp079: \n"
                            "fertile lherzolite : <0.7 GPa at 1000 °C, <0.8 GPa at 1100 °C \n"
                            "depleted lherzolite: above values shifted to lower pressure. "
                            "Length = nc. $\\text{GPa}$");
          prm.declare_entry("a_spl2plag", "5.0",
                            Patterns::List(Patterns::Double()),
                            "Multiplier factor increasing dTs/dDpl when entering plag field. "
                            "Length = nc. Units: None.");

        }
        prm.leave_subsection();
      }


      template <int dim>
      void
      Morgan2001MantleMelting<dim>::
      parse_parameters (ParameterHandler &prm)
      {
        prm.enter_subsection("Mantle melting settings");
        {
          nc = prm.get_double ("Number of lithologic components");
          melt_mode = prm.get ("Melting mode");
          equilibrate_Xs = prm.get_bool("Equilibrate solid water between components");
          Vol0 = Utilities::string_to_double(Utilities::split_string_list(prm.get("Initial volume fraction Vol0")));          
          Dpl0 = Utilities::string_to_double(Utilities::split_string_list(prm.get("Initial depletion Dpl0")));
          Xm0 = Utilities::string_to_double(Utilities::split_string_list(prm.get("Initial melt water Xm0")));
          Xb0 = Utilities::string_to_double(Utilities::split_string_list(prm.get("Initial bulk water Xb0")));
          Dm_H2O = Utilities::string_to_double(Utilities::split_string_list(prm.get("Water partition coefficient Dm_H2O")));
          Ds_H2O = Utilities::string_to_double(Utilities::split_string_list(prm.get("Water partition coefficient Ds_H2O")));
          f_trapped_max = Utilities::string_to_double(Utilities::split_string_list(prm.get("Maximum trapped melt volume fraction");
          Ts0 = Utilities::string_to_double(Utilities::split_string_list(prm.get("Solidus temperature at surface")));
          dTs_dP_intrinsic = Utilities::string_to_double(Utilities::split_string_list(prm.get("Solidus-pressure gradients")));
          dTs_dDpl_intrinsic = Utilities::string_to_double(Utilities::split_string_list(prm.get("Solidus-depletion gradients")));
          L0_equil = prm.get_double("Water diffusive length scale L0_equil");

          // Note that the latent heat of fusion dH = T * dS, dS is the entropy
          // of fusion. Negative because melting consumes heat. Here we use
          // a constant user_defined value of dH for each component. dH can 
          // also be calculated when dS is defined (not implemented yet).
          dH = Utilities::string_to_double(Utilities::split_string_list(prm.get("Latent heat of fusion")));

          // Phase boundary corrections
          include_cpx_out = prm.get_bool("Include cpx-out effect");
          include_plag_field = prm.get_bool("Include plag-to-spinel effect");
          Dpl_cpx_out = Utilities::string_to_double(Utilities::split_string_list(prm.get("Dpl_cpx_out")));
          a_cpx_out = Utilities::string_to_double(Utilities::split_string_list(prm.get("a_cpx_out")));
          P_spl2plag = Utilities::string_to_double(Utilities::split_string_list(prm.get("P_spl2plag")));
          a_spl2plag = Utilities::string_to_double(Utilities::split_string_list(prm.get("a_spl2plag")));
        }
        prm.leave_subsection();
      }
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    namespace ReactionModel
    {
#define INSTANTIATE(dim) \
  template class Morgan2001MantleMelting<dim>;

      ASPECT_INSTANTIATE(INSTANTIATE)
#undef INSTANTIATE
    }
  }
}
