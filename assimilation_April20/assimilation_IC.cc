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

#include "assimilation_IC.h"

#include <aspect/global.h>
#include <aspect/utilities.h>

#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/geometry_model/chunk.h>
#include <aspect/geometry_model/sphere.h>

#include <aspect/boundary_composition/box.h>
#include <aspect/boundary_composition/spherical_constant.h>
#include <aspect/boundary_composition/initial_composition.h>
#include <aspect/boundary_composition/interface.h>

#include <boost/lexical_cast.hpp>


namespace aspect
{
  namespace InitialComposition
  {
  template <int dim>
  Assimilation<dim>::Assimilation ()
  :
  current_file_number(0),
  first_data_file_model_time(0.0),
  first_data_file_number(0),
  data_file_time_step(0.0),
  time_weight(0.0),
  time_dependent(true),
  old_lookup(),
  lookup()
{}

  template <int dim>
  void
  Assimilation<dim>::initialize ()
  {
     //Check that the required geometry model (spherical shell, chunk, sphere) is used
	 AssertThrow ((Plugins::plugin_type_matches<GeometryModel::SphericalShell<dim> >(this->get_geometry_model()) ||
			       Plugins::plugin_type_matches<GeometryModel::Chunk<dim> >(this->get_geometry_model()) ||
   	               Plugins::plugin_type_matches<GeometryModel::Sphere<dim> >(this->get_geometry_model())),
	 ExcMessage ("This data assimilation plug-in currently is used for a spherical shell, chunk, sphere geometry."));

	 // Check that the required material model ("visco plastic") is used
	 //AssertThrow(Plugins::plugin_type_matches<MaterialModel::ViscoPlastic<dim> >(this->get_material_model()), \
	 //   	       ExcMessage("The assimilation plug-in requires the viscoplastic material model."));

	 //Check that the required initial temperature model (assimilation) is used
//	 AssertThrow(Plugins::plugin_type_matches<InitialTemperature::Assimilation<dim> >(this->get_initial_temperature_model()), \
//			     ExcMessage("The assimilation plug-in requires the initial temperature assimilation model."));

	 /* Call the Class: internal in assimilation_IT; add the IT head file in IC head file */
     lookup = std_cxx14::make_unique<InitialTemperature::internal::AssimilationLookup<dim>>(1, this->scale_factor);
     old_lookup = std_cxx14::make_unique<InitialTemperature::internal::AssimilationLookup<dim>>(1, this->scale_factor);

     // Set the first age-file number and load the first file
     current_file_number = first_data_file_number;

	 this->get_pcout() << std::endl << " Loading age assimilation data for the initial composition"
	                   << create_filename (current_file_number) << "." << std::endl << std::endl;

	 const std::string filename (create_filename (current_file_number));
	 const int next_file_number = (decreasing_file_order) ? current_file_number - 1 : current_file_number + 1;

	 if (Utilities::fexists(filename))
		 lookup->load_file(filename,this->get_mpi_communicator());
	 else
		 AssertThrow(false, ExcMessage (std::string("Age data file <") + filename + "> not found!"));

	 if (create_filename (current_file_number) == create_filename (current_file_number+1))
	 {
		 end_time_dependence ();
	 }
	 else
	 {
		 const std::string filename (create_filename (next_file_number));
	 	 this->get_pcout() << std::endl << " Update age data for prescribing composition "
	 	                   << filename << "." << std::endl << std::endl;

	 	 if (Utilities::fexists(filename))
	 	 {
	 		 lookup.swap(old_lookup);
	 	     lookup->load_file(filename,this->get_mpi_communicator());
	 	 }
	 	 else
	 		 end_time_dependence ();
	 }
  }

  template <int dim>
  std::string
  Assimilation<dim>::create_filename (const int timestep) const
  {
    std::string templ = data_directory+data_file_name;
    const int size = templ.length();
    std::vector<char> buffer(size+10);
    snprintf (buffer.data(), size + 10, templ.c_str(), timestep);
    std::string str_filename (buffer.data());
    return str_filename;
  }

  /* Update the age file */
  template <int dim>
  void
  Assimilation<dim>::update ()
  {
	  const double time_since_start = this->get_time() - first_data_file_model_time;

	  if (time_dependent && (time_since_start >= 0.0))
	  {
		  const int time_steps_since_start = static_cast<int> (time_since_start / data_file_time_step);
          this->get_pcout() << std::endl << " time steps since start "
	            	     	<< time_steps_since_start << "." << std::endl << std::endl;

          // whether we need to update our data files. This looks so complicated
	      // because we need to catch increasing and decreasing file orders and all
          // possible first_data_file_model_times and first_data_file_numbers.
	      const bool need_update = time_steps_since_start > std::abs(current_file_number - first_data_file_number);

	      if (need_update)
	      {
	        	// The last file, which was tried to be loaded was
	            // number current_file_number +/- 1, because current_file_number
	            // is the file older than the current model time
	            const int old_file_number =
	                  (decreasing_file_order)
	                  ?
	                  (current_file_number - 1)
	                  :
	                  (current_file_number + 1);

	            // Calculate new file_number
	            current_file_number =
	                  (decreasing_file_order)
	                  ?
	                  (first_data_file_number - time_steps_since_start)
	                  :
	                  (first_data_file_number + time_steps_since_start);

	            const bool load_both_files = std::abs(current_file_number - old_file_number) >= 1;
	            update_data(load_both_files);

	      }

	      time_weight = (time_since_start / data_file_time_step)
	                   - std::abs(current_file_number - first_data_file_number);

	      Assert ((0 <= time_weight) && (time_weight <= 1),
	      ExcMessage ("Error in set_current_time. Time_weight has to be in [0,1]"));
	  }
  }

  template <int dim>
  void
  Assimilation<dim>::update_data (const bool load_both_files)
  {
	  // If the time step was large enough to move forward more
      // then one data file we need to load both current files
      // to stay accurate in interpolation
      if (load_both_files)
      {
      	const std::string filename (create_filename (current_file_number));
        this->get_pcout() << std::endl << " Loading Age data assimilation file "
        		          "for prescribed lithospheric composition"
                          << filename << "." << std::endl << std::endl;
          if (Utilities::fexists(filename))
          {
          	lookup.swap(old_lookup);
            lookup->load_file(filename,this->get_mpi_communicator());
          }

          // If loading current_time_step failed, end
          // time dependent part with old_file_number.
          else
              end_time_dependence ();
      }

    // Now load the next data file. This part is the main purpose of this function.
    const int next_file_number =
              (decreasing_file_order) ?
              current_file_number - 1
              :
              current_file_number + 1;

    const std::string filename (create_filename (next_file_number));
    this->get_pcout() << std::endl << " Loading next age data assimilation file "
    		          "or prescribed lithospheric"
                      << filename << "." << std::endl << std::endl;
    if (Utilities::fexists(filename))
      {
        lookup.swap(old_lookup);
        lookup->load_file(filename,this->get_mpi_communicator());
      }

    // If next file does not exist, end time dependent part with current_time_step.
    else
      end_time_dependence ();
  }

  template <int dim>
  void
  Assimilation<dim>::end_time_dependence ()
  {
    // no longer consider the problem time dependent from here on out
    // this cancels all attempts to read files at the next time steps
    time_dependent = false;
    // Give warning if first processor
    this->get_pcout() << std::endl
            << " Loading new age data file did not succeed." << std::endl
            << " Assuming constant age-dependent conditions for the rest of the simulation "
            << std::endl << std::endl;
  }

  // Get the current plate age at each surface point
  template <int dim>
  double
  Assimilation<dim>::ascii_age (const Point<dim> &position) const
  {
    const std::array<double,dim> spherical_position =
	        Utilities::Coordinates::cartesian_to_spherical_coordinates(position);

	  Point<dim> internal_position = position;

    //Since we only consider the chunk/sphere,
	  for (unsigned int i = 0; i < dim; i++)
		  internal_position[i] = spherical_position[i];

	  if(!time_dependent)
	  {
		  return lookup->get_age(internal_position, 0);
	  }
	  else
	  {
		  // Get the old plate age at each surface point first
		  // then calculate the real age at current timestep
		  const double current_age = lookup->get_age(internal_position, 0);
		  const double old_age = old_lookup->get_age(internal_position,0);
		  const double real_age = time_weight * current_age + (1 - time_weight) * old_age;
		  return real_age;
	  }

  }

  template <int dim>
  double
  Assimilation<dim>::initial_composition (const Point<dim> &position, const unsigned int n_comp) const
  {

	  /* Get the depth of the point with respect to the reference surface */
  	  const double depth = this->get_geometry_model().depth(position);

      /* Get plate ages at each point */
      const double current_age = ascii_age(position);

      //From the experience of Gplates plugin
      const double magic_number = 1e-7 * this->get_geometry_model().maximal_depth();

      const std::array<double,dim> spherical_position =
    		Utilities::Coordinates::cartesian_to_spherical_coordinates(position);

      Point<dim> internal_position = position;
      for (unsigned int i = 0; i < dim; i++)
    	  internal_position[i] = spherical_position[i];

      if(simulator_access.introspection().compositional_name_exists("WZ"))
      {
    	  /* Add a weak boundary based on the coordinate*/
         //Another opition: define the weak zone from the input prm file
    	  const GeometryModel::Chunk<dim> &chunk_geometry =
    		  Plugins::get_plugin_as_type<const GeometryModel::Chunk<dim>> (this->get_geometry_model());

          const double lonmin = chunk_geometry.west_longitude ();
          const double lonmax = chunk_geometry.east_longitude ();
          const double latmin = chunk_geometry.south_latitude ();
          const double latmax = chunk_geometry.north_latitude ();
          const double wzlonmin = lonmin+wzs*numbers::PI/180;
          const double wzlonmax = lonmax-wzs*numbers::PI/180;
          const double wzlatmax = numbers::PI/2. - (latmin+wzs*numbers::PI/180);
          const double wzlatmin = numbers::PI/2. - (latmax-wzs*numbers::PI/180);

        	  if (internal_position[1]>=wzlonmin && internal_position[1]<=wzlonmax &&
        		  internal_position[2]>=wzlatmin && internal_position[2]<=wzlatmax)
        	  {
        		  // compo 0 = OC, Oceanic crust
        		  if  (current_age>0.001 && current_age< fixed_age_continent && depth <= d_oc+magic_number)
        		  {
        			  return (n_comp == 0 ) ? 1.0 : 0.0;
        		  }
        		  // compo 1 = OLM, Oceanic lithospheric mantle
        		  else if  (current_age>0.001 && current_age<fixed_age_continent
        		      		&& depth > d_oc+magic_number && depth <= d_oc+d_olm+magic_number)
        		  {
        			  return (n_comp == 1 ) ? 1.0 : 0.0;
        		  }
        		  // compo 2 = CC, Continental Crust
        		  else if (current_age>=fixed_age_continent && depth <= d_cc+magic_number)
        		  {
        			  return (n_comp == 2 ) ? 1.0 : 0.0;
        		  }
        		  // compo 3 = CLM, Continental lithospheric mantle
        		  else if (current_age>=fixed_age_continent && depth > d_cc+magic_number
        		      	   && depth <= d_cc+d_clm+magic_number)
        		  {
        			  return (n_comp == 3 ) ? 1.0 : 0.0;
        		  }
        		  // compo 4 = AS, Weak Asthenosphere
        		  else if ((current_age>=fixed_age_continent && depth > d_cc+d_clm+magic_number
        		      		&& depth <= d_tzt+magic_number) ||
        		      		(current_age>0 && current_age<fixed_age_continent
        		      		&& depth > d_oc+d_olm+magic_number && depth <= d_tzt+magic_number))
        		  {
        			  return (n_comp == 4 ) ? 1.0 : 0.0;
        		  }
        		  // compo 5 = TZ, Transition Zone
        		  else if (depth >= d_tzt+magic_number && depth <= d_tzb+magic_number)
        		  {
        			  return (n_comp == 5 ) ? 1.0 : 0.0;
        		  }
        		  // compo 6 = LM, Lower Mantle
        		  else if (depth >= d_tzb+magic_number)
        		  {
        			  return (n_comp == 6 ) ? 1.0 : 0.0;
        		  }
        		  else return 0;
        	  }
        	  else
        		  return (n_comp == 7 ) ? 1.0 : 0.0;	//Weak zone boundary

          }
          else
          {
        	  // compo 0 = OC, Oceanic crust
    		  if  (current_age>0.001 && current_age< fixed_age_continent && depth <= d_oc+magic_number)
    		  {
    			  return (n_comp == 0 ) ? 1.0 : 0.0;
    		  }
    		  // compo 1 = OLM, Oceanic lithospheric mantle
    		  else if  (current_age>0.001 && current_age<fixed_age_continent
    		      		&& depth > d_oc+magic_number && depth <= d_oc+d_olm+magic_number)
    		  {
    			  return (n_comp == 1 ) ? 1.0 : 0.0;
    		  }
    		  // compo 2 = CC, Continental Crust
    		  else if (current_age>=fixed_age_continent && depth <= d_cc+magic_number)
    		  {
    			  return (n_comp == 2 ) ? 1.0 : 0.0;
    		  }
    		  // compo 3 = CLM, Continental lithospheric mantle
    		  else if (current_age>=fixed_age_continent && depth > d_cc+magic_number
    		      	   && depth <= d_cc+d_clm+magic_number)
    		  {
    			  return (n_comp == 3 ) ? 1.0 : 0.0;
    		  }
    		  // compo 4 = AS, Weak Asthenosphere
    		  else if ((current_age>=fixed_age_continent && depth > d_cc+d_clm+magic_number
    		      		&& depth <= d_tzt+magic_number) ||
    		      		(current_age>0 && current_age<fixed_age_continent
    		      		&& depth > d_oc+d_olm+magic_number && depth <= d_tzt+magic_number))
    		  {
    			  return (n_comp == 4 ) ? 1.0 : 0.0;
    		  }
    		  // compo 5 = TZ, Transition Zone
    		  else if (depth >= d_tzt+magic_number && depth <= d_tzb+magic_number)
    		  {
    			  return (n_comp == 5 ) ? 1.0 : 0.0;
    		  }
    		  // compo 6 = LM, Lower Mantle
    		  else if (depth >= d_tzb+magic_number)
    		  {
    			  return (n_comp == 6 ) ? 1.0 : 0.0;
    		  }
    		  else return 0;
          }
  }

  template <int dim>
  void
  Assimilation<dim>::declare_parameters (ParameterHandler &prm)
  {
    prm.enter_subsection ("Initial composition model");
    {
      prm.enter_subsection("Age assimilation model");
      {
    	  prm.declare_entry ("Data directory",
    	        	         "$ASPECT_SOURCE_DIR/data/initial-composition/ascii-data/test/",
    	        	         Patterns::DirectoryName (),
    	        	         "The name of a directory that contains the age data. This path "
    	        	         "may either be absolute (if starting with a '/') or relative to "
    	        	         "the current directory. The path may also include the special "
    	        	         "text '$ASPECT_SOURCE_DIR' which will be interpreted as the path "
    	        	         "in which the ASPECT source files were located when ASPECT was "
    	        	         "compiled. This interpretation allows, for example, to reference "
    	        	         "files located in the `data/' subdirectory of ASPECT. ");
    	  prm.declare_entry ("Data file name", "assim_age_iniT.%d.dat",
    	        	         Patterns::Anything (),
    	        	         "The file name of the material data. Provide file in format: "
    	        	         "assim_age.\\%d.txt where \\%d is any sprintf integer "
    	        	         "qualifier, specifying the format of the current file number.");
    	  prm.declare_entry ("Data file time step", "1e6",
    	        	      	 Patterns::Double (0),
    	        	      	 "Time step between following age files. "
    	        	      	 "Depending on the setting of the global 'Use years in output instead of seconds' flag "
    	        	      	 "in the input file, this number is either interpreted as seconds or as years. "
    	        	      	 "The default is one million, i.e., either one million seconds or one million years.");
    	  prm.declare_entry ("Decreasing file order", "false",
    	        	         Patterns::Bool (),
    	                     "In some cases the boundary files are not numbered in increasing "
    	                     "but in decreasing order (e.g. 'Ma BP'). If this flag is set to "
      	                     "'True' the plugin will first load the file with the number "
    	                     "'First velocity file number' and decrease the file number during "
    	                     "the model run.");
    	  prm.declare_entry ("First data file model time", "0",
    	        	         Patterns::Double (0),
    	        	         "Time from which on the age file with number 'First data "
    	        	         "file number' is used as boundary condition. Previous to this "
    	        	         "time, a no-slip boundary condition is assumed. Depending on the setting "
    	        	         "of the global 'Use years in output instead of seconds' flag "
    	        	         "in the input file, this number is either interpreted as seconds or as years.");
    	  prm.declare_entry ("First data file number", "0",
    	        	         Patterns::Integer (),
    	        	         "Number of the first age file to be loaded when the model time "
    	        	         "is larger than 'First data file model time'.");
    	  prm.declare_entry ("Scale factor", "1",
    	        	         Patterns::Double (),
    	        	         "Scalar factor, which is applied to the boundary velocity. "
    	        	         "You might want to use this to scale the velocities to a "
    	        	         "reference model (e.g. with free-slip boundary) or another "
    	        	         "plate reconstruction.");
    	  prm.declare_entry ("Oceanic crustal thickness", "100000.0",
      	                     Patterns::Double (0),
      	                     "The thickness of the oceanic crust. Units: m. ");
    	  prm.declare_entry ("Oceanic lithospheric mantle thickness", "60000.0",
      	                     Patterns::Double (0),
							 "The thickness of the oceanic lithospheric mantle. Units: m. ");
    	  prm.declare_entry ("Continental crustal thickness", "40000.0",
      	                     Patterns::Double (0),
      	                     "The thickness of the continental crust. Units: m. ");
    	  prm.declare_entry ("Continental lithospheric mantle thickness", "110000.0",
      	                     Patterns::Double (0),
							 "The thickness of the continental lithospheric mantle. Units: m. ");
    	  prm.declare_entry ("Transition zone top", "410000.0",
    	      	             Patterns::Double (0),
    						 "The top depth of the transition zone. Units: m. ");
		  prm.declare_entry ("Transition zone bottom", "660000.0",
					    	 Patterns::Double (0),
					    	 "The bottom depth of the transition zone. Units: m. ");
    	  prm.declare_entry ("Weak zone size", "5.0",
      	                     Patterns::Double (0),
							 "The size of weak zone in the boundary. Units: degree. ");
    	  prm.declare_entry ("Age_continent", "200",
    	                     Patterns::Double (0),
    	                     "The plate with the age above this value is assumed to be the continent. "
    						 "Unit: Myr");

      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  template <int dim>
  void
  Assimilation<dim>::parse_parameters (ParameterHandler &prm)
   {
      AssertThrow (dim == 3, ExcMessage ("The ' Data assimilation' model for the initial "
  	               "composition is only available for 3d sphere/chunk computations currently."));

  	  prm.enter_subsection ("Initial composition model");
  	  {
  		prm.enter_subsection("Age assimilation model");
  		{
		       data_directory = Utilities::expand_ASPECT_SOURCE_DIR(prm.get ("Data directory"));
		       data_file_name = prm.get ("Data file name");
		       data_file_time_step        = prm.get_double ("Data file time step");
		       first_data_file_model_time = prm.get_double ("First data file model time");
		       first_data_file_number     = prm.get_integer("First data file number");
		       decreasing_file_order      = prm.get_bool   ("Decreasing file order");
		       scale_factor               = prm.get_double ("Scale factor");

  			   d_oc = prm.get_double ("Oceanic crustal thickness");
               d_olm = prm.get_double ("Oceanic lithospheric mantle thickness");
               d_cc = prm.get_double ("Continental crustal thickness");
               d_clm = prm.get_double ("Continental lithospheric mantle thickness");
               d_tzt = prm.get_double ("Transition zone top");
               d_tzb = prm.get_double ("Transition zone bottom");
               wzs = prm.get_double ("Weak zone size");
               fixed_age_continent = prm.get_double ("Age_continent");

               if (this->convert_output_to_years()==true)
               {
            	   data_file_time_step        *= year_in_seconds;
                   first_data_file_model_time *= year_in_seconds;
               }
  		prm.leave_subsection ();
  	    }
  	  prm.leave_subsection ();
	  }
   }
  }
}

  // explicit instantiations
namespace aspect
{
  namespace InitialComposition
  {
      ASPECT_REGISTER_INITIAL_COMPOSITION_MODEL(Assimilation,
                                              "age_assimilation_C",
                                              "Implementation of a model in which the initial composition "
                                              "is derived from age files from the GPlates program.")
  }
}
