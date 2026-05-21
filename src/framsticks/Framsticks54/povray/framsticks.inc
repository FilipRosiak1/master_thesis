// Use one of the framsticks-*.inc file names below. Each of them defines a different style and scenery, and supports different settings.
#include "framsticks-standard.inc"
//#include "framsticks-autumn.inc"
//#include "framsticks-matrix.inc"
//#include "framsticks-robo.inc"
//#include "framsticks-robo2.inc"
//#include "framsticks-whitevoid.inc"
//#include "framsticks-forams-example.inc"

//optional settings:

//#declare LightAz=90;
//#declare SunPhase=0.5;
//#declare LightAlt=5;
//#declare LightSize=1000; //area_light

//#declare Override_OFFSET_Z=-1;
//#declare HeightfieldMode=...;//0=Triangles,1=Smooth triangles,2=Bicubic patch,3=Bicubic patch alternate mode
//#declare HeightfieldSubdivision=0;
//#declare BumpyGround=true; //only works for flat world, does not blend seamlessly with the infinite plane
//#declare BumpyGroundScale=10; //size of a single BumpyGround tile
//#declare BumpyGroundHeight=0.2; //vertical scaling of the BumpyGround
//#declare SandTextureScale=0.05; //fine grained luxury sand!

//#declare AirFogDistance = 200;
//#declare WaterFogColor = rgbf<0.309, 0.694, 0.952, 0.3>;
//#undef WaterFogColor //disable underwater fog when underwater media is used
//we might want to adjust the fog distance depending on the world size:
//#declare WaterFogDistance = 20;
//alternative seabottom texture:
//#declare UnderwaterTexture=texture { pigment { granite color_map { [ 0 rgb <0.227, 0.533, 0.250> ] [ 1 rgb <0.682, 0.749, 0.650> ] } scale 2 } finish {ambient 0.7/GlobalAmbient diffuse 0.3 } }

//#declare ForamIOR=1.0; //should be used when rendering live foraminifera not submerged in water
//#declare DeadForamTextureScale=1.0; //adjusting foraminifera texture scale
//#declare ForceDeadForamTexture=true; //make all foraminifera use the dead texture (also speeds up rendering 1000x because no media is used)

// animated sun+sky
//#if (AnimFrame<150)
//  #declare SunPhase=AnimFrame/150;
//  #declare LightAlt=50*AnimFrame/150; //default is 50 (framsticks-standard.inc)
//#end

//#include "particles.inc"
//#macro EndSceneExtra()
//tiny, slowly moving random particles in water
//RandomParticles(2500,<0,0,0>,<field_World_wrldsiz,field_World_wrldsiz,field_World_wrldsiz>,0.2,10,0.01)
//more realistic underwater light scattering using media:
//cylinder { <field_World_wrldsiz/2,field_World_wrldsiz/2,OFFSET_Z-0.5>,<field_World_wrldsiz/2,field_World_wrldsiz/2,field_World_wrldwat+OFFSET_Z-0.001>,3*field_World_wrldsiz  hollow no_shadow
//  texture { pigment { rgbt 1 } } finish { ambient 0 diffuse 0 }
//  interior { ior WaterIOR media { intervals 10 scattering { 2,<1.0,0.8,0.7>*0.02 extinction 0.04 } absorption 0.02*<1.0,0.6,0.4> } } }
//#end

//#include "framsticks-skijump.inc"

//Camera(<36.83832931518555, 7.299591064453125, 15.460891723632812>,<30.84323655953631, 28.264855206012726, 11.199661940336227>) //force camera position/direction
//#declare StandardCamera=camera { spherical up <0,0,1> right <1,0,0> sky <0,0,-1> location <0,0,0> look_at <0,1,0> } //spherical 360 camera
//#declare CameraIgnoreRotation=true;

//#macro PartGeometry(Location,Rotation)
//  PartGeometryStd(Location,Rotation) //call original visualization (or not)
//  #if something // add custom elements (or not)
//  sphere { <0,0,0>,1 texture { pigment { color Red } } rotate Rotation*57.3 translate Location }
//  #end
//#end
