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

#ifndef _aspect_material_model_reaction_model_morgan2001_mantle_melting_h
#define _aspect_material_model_reaction_model_morgan2001_mantle_melting_h

#include <aspect/material_model/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/postprocess/melt_statistics.h>
#include <aspect/melt.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace ReactionModel
    {

      /**
       * A multicomponent, water-sensitive mantle melting model based on Morgan (2001).
       *
       * @ingroup ReactionModel
       */
      template <int dim>
      class Morgan2001MantleMelting : public ::aspect::SimulatorAccess<dim>
      {
        public:
          /**
           * Initialize compositional field indices for melt-related variables.
           */
          void
          initialize ();

          /**
           * Declare the parameters this function takes through input files.
           */
          static
          void
          declare_parameters (ParameterHandler &prm);

          /**
           * Read the parameters from the parameter file.
           */
          void
          parse_parameters (ParameterHandler &prm);

          /**
           * Compute all the reaction term variables needed for the melting model
           * based on the Morgan 2001 formulation.
           */
          void fill_reaction_outputs (const typename Interface<dim>::MaterialModelInputs &in,
                                      typename Interface<dim>::MaterialModelOutputs &out) const;

        private:
          void
          compute_current_solidus (const double              P_GPa_positive,
                                   const std::vector<double> &Dpl,
                                   const std::vector<double> &Xm,
                                   std::vector<double>       &Ts_dry,
                                   std::vector<double>       &Ts_wet,
                                   std::vector<double>       &Ts_liquidus) const;

          double
          melt_water_saturation (const double pressure_GPa) const;

          double
          modify_dTs_dDpl_cpx_plg (const unsigned int i,
                                   const double pressure_GPa,
                                   const std::vector<double> &Dpl) const;

          void
          compute_batch_water_partitioning (const std::vector<double> &F,
                                            std::vector<double>       &Xs,
                                            std::vector<double>       &Xm,
                                            std::vector<double>       &dXs_dF,
                                            std::vector<double>       &dXm_dF) const;

          void
          compute_fractional_water_partitioning (const std::vector<double> &F,
                                                 std::vector<double>       &Xs,
                                                 std::vector<double>       &Xm,
                                                 std::vector<double>       &dXs_dF,
                                                 std::vector<double>       &dXm_dF) const;

          void
          compute_incremental_water_partitioning (const std::vector<double> &f_trapped,
                                                  const std::vector<double> &Xb,
                                                  std::vector<double>       &Xs,
                                                  std::vector<double>       &Xm,
                                                  std::vector<double>       &dXs_dF,
                                                  std::vector<double>       &dXm_dF) const;

          std::vector<double>
          equilibrate_water_between_solids (const std::vector<double> &Vol,
                                            const std::vector<double> &f_trapped,
                                            const std::vector<double> &Xs) const;

          std::vector<double>
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
                                  std::vector<double>       &dXs_dF_equil) const;
          /**
           * Parameters for hydrous melting of multicomponent mantle after Morgan, 2001
           */
          unsigned int nc;
          std::string melt_mode;
          bool equilibrate_Xs;
          std::vector<double> Vol0;
          std::vector<double> Dpl0;
          std::vector<double> Xm0;
          std::vector<double> Xb0;
          std::vector<double> Dm_H2O;
          std::vector<double> Ds_H2O;
          std::vector<double> f_trapped_max;
          std::vector<double> Ts0;
          std::vector<double> dTs_dP_intrinsic;
          std::vector<double> dTs_dDpl_intrinsic;
          std::vector<double> dH;          
          double L0_equil;
          bool include_cpx_out;
          bool include_plag_field;
          std::vector<double> Dpl_cpx_out;
          std::vector<double> a_cpx_out;
          std::vector<double> P_spl2plag;
          std::vector<double> a_spl2plag;

          std::vector<unsigned int> F_field_indices;
          std::vector<unsigned int> f_trapped_field_indices;
          std::vector<unsigned int> Xb_field_indices;
          std::vector<unsigned int> Vol_field_indices;
      };
    }

  }
}

#endif
