//Adds N tiny, slowly moving, bright random particles in water.
//Examples:   https://www.youtube.com/watch?v=RGOAV0pfzpI   and   https://www.youtube.com/watch?v=2JrV2NmYu3U

#macro Additive_Blend(Color, Additiveness, Transparency)
	// (Can't have Additiveness == 1 as the hackish nature of this will cause division by zero)
	#if (Additiveness > 0.99999 & Additiveness <= 1.0)
		#local Additiveness = 0.99999;
	#end
	#if (Additiveness > 1.0 & Additiveness < 1.00001)
		#local Additiveness = 1.00001;
	#end
	
	#local Color = <1,1,1> * Color / (1 - Additiveness);
	rgbt <Color.x, Color.y, Color.z, Transparency + (1 - Transparency) * Additiveness>
#end

// Particles move chaotically in a box.
//
// N = number of particles
// Center_xyz, R_xyz = particles are created in the box centered at Center_xyz, edge helf-length is R_xyz
// Size = particle size
// Movement = radius of a sphere where a particle chaotically moves
// Speed = base speed of particles (in RADIANS) per frame
//
// reasonable parameters: RandomParticles(1000,<10,10,10>,<20,20,20>,0.5,5,0.01)
#macro RandomParticles(N,Center_xyz,R_xyz,Size,Movement,Speed)
 #local R=seed(12345);
 #local i=0;
 #while (i<N)
  #local P=Center_xyz+R_xyz*<2*rand(R)-1,2*rand(R)-1,2*rand(R)-1>; //select a random location in the box
  #local P=P+<sin(rand(R)*2*pi+AnimFrame*Speed*(rand(R)+1))*Movement,sin(rand(R)*2*pi+AnimFrame*Speed*(rand(R)+1))*Movement,sin(rand(R)*2*pi+AnimFrame*Speed*(rand(R)+1))*Movement>;
  #local OK=0;
  #ifndef (ParticlesInside)
     #local OK=1;
  #else
     #local OK=inside(ParticlesInside,P);
  #end
  #if (OK)
    /*
  cylinder
  {
   <0,-0.1,0>,<0,0.1,0>,0.999 no_shadow
   pigment { spherical color_map { [0 Additive_Blend(<0.001,0.001,0.001>,1,0) ] [1 Additive_Blend(<0.003,0.003,0.003>,1,0) ] } }
   finish { ambient 10 diffuse 0 }
   interior { ior WaterIOR }
   scale (rand(R)+1)*0.5*Size
   rotate LookAtRotation(CameraLocation-P)
   translate P
  }
  #end
  #local i=i+1;
 #end
#end
