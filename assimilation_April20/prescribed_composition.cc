/*
  Copyright (C) 2011 - 2019 by the authors of the ASPECT code.

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

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/fe/fe_values.h>
#include <aspect/global.h>
#include <aspect/simulator_signals.h>
#include <aspect/geometry_model/interface.h>
#include <aspect/initial_composition/interface.h>

#include <aspect/geometry_model/chunk.h>
#include "assimilation_IC.h"
#include <aspect/utilities.h>

namespace aspect
{
  using namespace dealii;

  // Global variables (to be set by parameters)
  bool prescribe_internal_composition;
  double fixed_OLAB_depth;
  double fixed_CLAB_depth;
  double fixed_age_continent;
  double wzs;

  /**
   * Declare additional parameters.
   */
  void declare_parameters(const unsigned int dim,
                          ParameterHandler &prm)
  {
    prm.declare_entry ("Prescribe internal composition", "false",
                       Patterns::Bool (),
                       "Whether or not to use any prescribed internal composition. "
                       "Currently, locations in which to prescribe compositions "
                       "are defined in the lithosphere ");
    prm.declare_entry ("OLAB depth", "80000",
    		           Patterns::Double (0),
                       "The depth of the lithosphere-asthenosphere boundary for "
                       "the oceanic plate. ");
    prm.declare_entry ("CLAB depth", "150000",
    		           Patterns::Double (0),
                       "The depth of the lithosphere-asthenosphere boundary for "
                       "the continental plate. ");
    prm.declare_entry ("Age_continent", "200",
                       Patterns::Double (0),
                       "The plate with the age above this value is assumed to be the continent. "
					   "Unit: Myr");
	prm.declare_entry ("Weak zone size", "5.0",
	                   Patterns::Double (0),
					   "The size of weak zone in the boundary. Units: degree. ");
   }

  template <int dim>
    void parse_parameters(const Parameters<dim> &,
                          ParameterHandler &prm)
    {
      prescribe_internal_composition = prm.get_bool ("Prescribe internal composition");
      fixed_OLAB_depth = prm.get_double ("OLAB depth");
      fixed_CLAB_depth = prm.get_double ("CLAB depth");
      fixed_age_continent = prm.get_double ("Age_continent");
      wzs = prm.get_double ("Weak zone size");

    }

  template <int dim>
  void constrain_internal_composition (const SimulatorAccess<dim> &simulator_access,
                                          ConstraintMatrix &current_constraints)
                                          //Go to the current constraints
  {
      if (prescribe_internal_composition)
        {
    	  // This is to point to the specific the class of Assimilation under
    	  // Namespace: InitialComposition for the plugin assimilation_IC
          const InitialComposition::Assimilation <dim> &const_age_assim
                             = Plugins::get_plugin_as_type<const InitialComposition::Assimilation <dim> >
                             (simulator_access.get_initial_composition());
          InitialComposition::Assimilation <dim> &age_assim =
        		  const_cast <InitialComposition::Assimilation <dim> &> (const_age_assim);

          // Call the update funcition
          age_assim.update();

          //get the support points;the quadrature points.
          const std::vector< Point<dim> > points = simulator_access.get_fe().get_unit_support_points();
          const Quadrature<dim> quadrature (points);
          //FEValues<dim> fe_values (simulator_access.get_fe(), quadrature, update_quadrature_points | update_values);
          FEValues<dim> fe_values (simulator_access.get_fe(), quadrature, update_quadrature_points);
          typename DoFHandler<dim>::active_cell_iterator cell;

          // Loop over all cells
          //Find the active part
          for (cell = simulator_access.get_dof_handler().begin_active();
               cell != simulator_access.get_dof_handler().end(); ++cell)
          {
            if (! cell->is_artificial())		//Find the local MPI processor
              {
                fe_values.reinit (cell);	//Find the coordinate values of the current cell
                //std::vector<unsigned int> local_dof_indices(simulator_access.get_fe().dofs_per_cell);
                std::vector<types::global_dof_index> local_dof_indices(simulator_access.get_fe().dofs_per_cell);
                cell->get_dof_indices (local_dof_indices);

                for (unsigned int q=0; q<quadrature.size(); q++)
                {
                	// If it's okay to constrain this DOF;
                	//exclude those already constrained cells (e.g., boundary points and some interaction points)
                  if (current_constraints.can_store_line(local_dof_indices[q]) &&
                      !current_constraints.is_constrained(local_dof_indices[q]))
                    {

                	  // Get the composition component index  (0 - n; n = 3(v)+2(T,P)+compo num)
                      const unsigned int c_idx =
                    		  simulator_access.get_fe().system_to_component_index(q).first;

                      // If we're on composition DOFs
                      std::vector<unsigned int> cfield =
                    		  simulator_access.introspection().component_indices.compositional_fields;
                      //const unsigned int id_w = simulator_access.introspection().compositional_index_for_name("WZ");

                      //Get the point position
                	  const Point<dim> p = fe_values.quadrature_point(q);

                      // Get the depth and the age for each point
                      const double depth = simulator_access.get_geometry_model().depth(p);
                      const double current_age = age_assim.ascii_age(p);

if (depth <= 200e3)
{                      simulator_access.get_pcout() << c_idx << " " << current_age << " " << p
                                          << " " << depth << " " << std::endl;
}
                      //From the experience of Gplates plugin
                      const double magic_number = 1e-7 * simulator_access.get_geometry_model().maximal_depth();

                      // Update the compositional fields above the LAB due to the age data assimilation

                      if ((c_idx >= cfield[0]) && (c_idx <= cfield[cfield.size()-1]))
                      {
                    	  //If the chunk has a weak zone boundary
                    	  if(simulator_access.introspection().compositional_name_exists("WZ"))
                    	  {
                    	    //Get the spherical position
                    	    const std::array<double,dim> spherical_position =
                          		Utilities::Coordinates::cartesian_to_spherical_coordinates(p);

                            Point<dim> internal_position = p;
                            for (unsigned int i = 0; i < dim; i++)
                          	  internal_position[i] = spherical_position[i];

                            const GeometryModel::Chunk<dim> &const_chunk_geometry =
                          		  Plugins::get_plugin_as_type<const GeometryModel::Chunk<dim>>
								  (simulator_access.get_geometry_model());

                          	GeometryModel::Chunk<dim> &chunk_geometry =
                          			const_cast <GeometryModel::Chunk<dim> &> (const_chunk_geometry);

                            const double lonmin = chunk_geometry.west_longitude ();
                            const double lonmax = chunk_geometry.east_longitude ();
                            const double latmin = chunk_geometry.south_latitude ();
                            const double latmax = chunk_geometry.north_latitude ();
                            const double wzlonmin = lonmin+wzs*numbers::PI/180;
                            const double wzlonmax = lonmax-wzs*numbers::PI/180;
                            const double wzlatmax = numbers::PI/2. - (latmin+wzs*numbers::PI/180);
                            const double wzlatmin = numbers::PI/2. - (latmax-wzs*numbers::PI/180);

                            if (internal_position[1]>=wzlonmin && internal_position[1]<=wzlonmax &&
                        	internal_position[2]>=wzlatmin && internal_position[2]<=wzlatmax &&
                                ((current_age>0.001 && current_age<fixed_age_continent &&
                                   depth <= (fixed_OLAB_depth + magic_number)) ||
                                 (current_age> fixed_age_continent && depth <= (fixed_CLAB_depth + magic_number)))
                               )
                            {
                            	//if((current_age>0.001 && current_age<fixed_age_continent &&
                            	//    depth <= (fixed_OLAB_depth + magic_number)) ||
				//   (current_age> fixed_age_continent && depth <= (fixed_CLAB_depth + magic_number))
				//  )
                                //{
                                    if(age_assim.initial_composition(p,c_idx-5)==1)
                                  	  simulator_access.get_pcout() << c_idx << " " << current_age
					  << " " << depth << " " << internal_position << " "
					  << age_assim.initial_composition(p,c_idx-5) << std::endl;

                                    current_constraints.add_line (local_dof_indices[q]);
                                    current_constraints.set_inhomogeneity (local_dof_indices[q], age_assim.initial_composition(p,(c_idx-5)));
                                //}
                            }
                    	  }
                    	  else
                    	  {
                    		  if((current_age>0.001 && current_age<fixed_age_continent &&
                    		      depth <= (fixed_OLAB_depth + magic_number)) ||
                    		     (current_age> fixed_age_continent && depth <= (fixed_CLAB_depth + magic_number))
                    		    )
                    		  {
                    		      current_constraints.add_line (local_dof_indices[q]);
                    		      current_constraints.set_inhomogeneity (local_dof_indices[q], age_assim.initial_composition(p,(c_idx-5)));
                    		  }
                    	  }
                      }
                    }
                }
              }
          }
        }
  }

  // Connect declare_parameters and parse_parameters to appropriate signals.
   void parameter_connector ()
   {
     SimulatorSignals<2>::declare_additional_parameters.connect (&declare_parameters);
     SimulatorSignals<3>::declare_additional_parameters.connect (&declare_parameters);

     SimulatorSignals<2>::parse_additional_parameters.connect (&parse_parameters<2>);
     SimulatorSignals<3>::parse_additional_parameters.connect (&parse_parameters<3>);
   }

   // Connect constraints function to correct signal.
   template <int dim>
   void signal_connector (SimulatorSignals<dim> &signals)
   {
     signals.post_constraints_creation.connect (&constrain_internal_composition<dim>);
   }

   // Tell Aspect to send signals to the connector functions
   ASPECT_REGISTER_SIGNALS_PARAMETER_CONNECTOR(parameter_connector)
   ASPECT_REGISTER_SIGNALS_CONNECTOR(signal_connector<2>, signal_connector<3>)
 }
