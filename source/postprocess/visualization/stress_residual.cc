/*
  Copyright (C) 2011 - 2021 by the authors of the ASPECT code.

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


#include <aspect/postprocess/visualization/stress_residual.h>
#include <aspect/material_model/visco_plastic.h>

namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      template <int dim>
      StressResidual<dim>::
      StressResidual ()
        :
        DataPostprocessorScalar<dim> ("stress_residual",
                                      update_values | update_gradients | update_quadrature_points)
      {}



      template <int dim>
      void
      StressResidual<dim>::
      evaluate_vector_field(const DataPostprocessorInputs::Vector<dim> &input_data,
                            std::vector<Vector<double>> &computed_quantities) const
      {
        AssertThrow(Plugins::plugin_type_matches<const MaterialModel::ViscoPlastic<dim>>(this->get_material_model()),
                    ExcMessage("This postprocessor only works with the viscoplastic material model. "));

        const unsigned int n_quadrature_points = input_data.solution_values.size();
        Assert(computed_quantities.size() == n_quadrature_points, ExcInternalError());
        Assert(computed_quantities[0].size() == 1, ExcInternalError());
        Assert(input_data.solution_values[0].size() == this->introspection().n_components, ExcInternalError());
        Assert(input_data.solution_gradients[0].size() == this->introspection().n_components, ExcInternalError());

        double dtc = this->get_timestep();

        MaterialModel::MaterialModelInputs<dim> in(input_data,
                                                   this->introspection());
        MaterialModel::MaterialModelOutputs<dim> out(n_quadrature_points,
                                                     this->n_compositional_fields());

        this->get_material_model().create_additional_named_outputs(out);

        in.requested_properties = MaterialModel::MaterialProperties::viscosity | MaterialModel::MaterialProperties::additional_outputs;
        in.current_cell = input_data.template get_cell<dim>();

        // Get the viscosity
        this->get_material_model().evaluate(in, out);

        for (unsigned int q = 0; q < n_quadrature_points; ++q)
          {
            const SymmetricTensor<2, dim> strain_rate = in.strain_rate[q];
            const SymmetricTensor<2, dim> deviatoric_strain_rate = (this->get_material_model().is_compressible()
                                                                    ? strain_rate - 1. / 3 * trace(strain_rate) * unit_symmetric_tensor<dim>()
                                                                    : strain_rate);

            const double eta = out.viscosities[q];

            // Compressive stress is positive in geoscience applications
            SymmetricTensor<2, dim> stress = -2. * eta * deviatoric_strain_rate +
                                             in.pressure[q] * unit_symmetric_tensor<dim>();

            if (this->get_parameters().enable_elasticity == true)
              {
                // Visco-elastic stresses are stored on the fields
                SymmetricTensor<2, dim> stress_0, stress_old;
                stress_0[0][0] += in.composition[q][this->introspection().compositional_index_for_name("ve_stress_xx")];
                stress_0[1][1] += in.composition[q][this->introspection().compositional_index_for_name("ve_stress_yy")];
                stress_0[0][1] += in.composition[q][this->introspection().compositional_index_for_name("ve_stress_xy")];

                stress_old[0][0] = in.composition[q][this->introspection().compositional_index_for_name("ve_stress_xx_old")];
                stress_old[1][1] = in.composition[q][this->introspection().compositional_index_for_name("ve_stress_yy_old")];
                stress_old[0][1] = in.composition[q][this->introspection().compositional_index_for_name("ve_stress_xy_old")];

                if (dim == 3)
                  {
                    stress_0[2][2] += in.composition[q][this->introspection().compositional_index_for_name("ve_stress_zz")];
                    stress_0[0][2] += in.composition[q][this->introspection().compositional_index_for_name("ve_stress_xz")];
                    stress_0[1][2] += in.composition[q][this->introspection().compositional_index_for_name("ve_stress_yz")];

                    stress_old[2][2] = in.composition[q][this->introspection().compositional_index_for_name("ve_stress_zz_old")];
                    stress_old[0][2] = in.composition[q][this->introspection().compositional_index_for_name("ve_stress_xz_old")];
                    stress_old[1][2] = in.composition[q][this->introspection().compositional_index_for_name("ve_stress_yz_old")];
                  }

                const MaterialModel::ElasticAdditionalOutputs<dim> *elastic_out = out.template get_additional_output<MaterialModel::ElasticAdditionalOutputs<dim>>();

                const double shear_modulus = elastic_out->elastic_shear_moduli[q];

                // $\eta_{el} = G \Delta t_{el}$
                double elastic_timestep = this->get_timestep();
                double elastic_viscosity = elastic_timestep * shear_modulus;
                if (Plugins::plugin_type_matches<MaterialModel::ViscoPlastic<dim>>(this->get_material_model()))
                  {
                    const MaterialModel::ViscoPlastic<dim> &vp = Plugins::get_plugin_as_type<const MaterialModel::ViscoPlastic<dim>>(this->get_material_model());
                    elastic_viscosity = vp.get_elastic_viscosity(shear_modulus);
                    elastic_timestep = vp.get_elastic_timestep();
                  }
                else if (Plugins::plugin_type_matches<MaterialModel::Viscoelastic<dim>>(this->get_material_model()))
                  {
                    const MaterialModel::Viscoelastic<dim> &ve = Plugins::get_plugin_as_type<const MaterialModel::Viscoelastic<dim>>(this->get_material_model());
                    elastic_viscosity = ve.get_elastic_viscosity(shear_modulus);
                    elastic_timestep = ve.get_elastic_timestep();
                  }
                else
                  AssertThrow(false, ExcMessage("The stress residual postprocessor cannot be used with elasticity for material models other than ViscoPlastic and Viscoelastic."));

                if (dtc == 0 && this->get_timestep_number() == 0)
                  dtc = std::min(std::min(this->get_parameters().maximum_time_step, this->get_parameters().maximum_first_time_step), elastic_timestep);
                const double timestep_ratio = dtc / elastic_timestep;
                // Scale the elastic viscosity with the timestep ratio, eta is already scaled.
                elastic_viscosity *= timestep_ratio;

                // Apply the stress update to get the total stress of timestep t.
                stress = 2. * eta * deviatoric_strain_rate + eta / elastic_viscosity * stress_0 + (1. - timestep_ratio) * (1. - eta / elastic_viscosity) * stress_old;
              }

            // Compute the deviatoric stress
            const SymmetricTensor<2, dim> deviatoric_stress = deviator(stress);

            // Compute the second moment invariant of the deviatoric stress
            const double stress_invariant = std::sqrt(std::fabs(second_invariant(deviatoric_stress)));

            // Get the current yield_stress
            const MaterialModel::PlasticAdditionalOutputs<dim> *plastic_output =
              out.template get_additional_output<MaterialModel::PlasticAdditionalOutputs<dim>>();
            const double yield_stress = plastic_output->yield_stresses[q];

            // Compute the difference between the second stress invariant and the yield stress
            computed_quantities[q](0) = stress_invariant - yield_stress;
          }

        // average the values if requested
        const auto &viz = this->get_postprocess_manager().template get_matching_postprocessor<Postprocess::Visualization<dim>>();
        if (!viz.output_pointwise_stress_and_strain())
          average_quantities(computed_quantities);
      }
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    namespace VisualizationPostprocessors
    {
      ASPECT_REGISTER_VISUALIZATION_POSTPROCESSOR(StressResidual,
                                                  "stress residual",
                                                  "A visualization output object that generates output "
                                                  "for the difference between the second moment invariant "
                                                  "of the deviatoric stress tensor and the yield stress. "
                                                  "Note that this plugin currently only works when the "
                                                  "'viscoplastic' material model is used. ")
    }
  }
}
