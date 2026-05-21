//
// framsticks.inc, standard (default) setup
//
#include "colors.inc"
#include "textures.inc"
#include "woods.inc"
#include "stones.inc"
#include "foram.inc"
#include "functions.inc"

#debug concat("Animation Frame #",str(AnimFrame,5,0))

////////////////////////////////
// you can override these settings in your "framsticks-user.inc" file:

#declare OFFSET_Z=0.0;

#declare LightAz=60;
#declare LightAlt=50;
#declare LightDistance=4000;
#declare LightSize=300;// >0 means area_light
#declare HeightfieldMode=2;//0=Triangles,1=Smooth triangles,2=Bicubic patch,3=Bicubic patch alternate mode
// #declare Override_OFFSET_Z=...;
#declare HeightfieldSubdivision=0;
#declare WaterFogColor = rgbf<0.1, 0.4, 0.6, 0.3>;
#declare WaterFogDistance = 20;
#declare UnderwaterSurfaceGlow = <0.9,1,1,0.9>;
#declare WaterTrueWaves = 0;
#declare WaterTrueWavesAmp=1;
#declare WetSandAboveWater = 0.2;
#declare AirFogDistance = 200;
#declare AirFogThickness = 5;
#declare EnergyRadiusMax = 1.0;
#declare EnergyRadiusMin = 1.0;
#declare EnergyLightShadows = false;
#declare EnergyMediaInteraction=false;

#declare FORAM_M_VSTYLE="foram";

#declare SkyBrightness=1.0;
#declare HorizonColor_2=<0.9, 0.92, 0.95>;
#declare MidSkyColor0_2=<0.93,0.93,0.93>;
#declare MidSkyColor1_2=<0.88,0.88,0.88>;
#declare MidSkyColor2_2=<0.3, 0.3, 0.8>;
#declare MidSkyColor3_2=<0.23, 0.29, 0.93>;
#declare HighSkyColor0_2=<0.95,0.95,0.95>;
#declare HighSkyColor1_2=<0.9,0.9,0.9>;
#declare HighSkyColor2_2=<0.3, 0.3, 0.8>;
#declare HighSkyColor3_2=<0.04, 0.16, 0.66>;
#declare LightColor_2=<1,1,1>;

#declare HorizonColor_1=<0.9, 0.8, 0.6>;
#declare MidSkyColor0_1=<0.93,0.93,0.43>;
#declare MidSkyColor1_1=<0.88,0.78,0.48>;
#declare MidSkyColor2_1=<0.8, 0.5, 0.5>;
#declare MidSkyColor3_1=<0.4, 0.2, 0.5>;
#declare HighSkyColor0_1=<0.9,0.9,0.9>;
#declare HighSkyColor1_1=<0.9,0.5,0.8>;
#declare HighSkyColor2_1=<0.6, 0.3, 0.6>;
#declare HighSkyColor3_1=<0.3, 0.2, 0.5>;
#declare LightColor_1=<1,0.7,0.3>;

#declare HorizonColor_0=<0.9, 0.4, 0.2>;
#declare MidSkyColor0_0=<0.9,0.1,0.7>;
#declare MidSkyColor1_0=<0.7,0.5,0.4>;
#declare MidSkyColor2_0=<0.3,0.2,0.6>;
#declare MidSkyColor3_0=<0.3,0.05,0.3>;
#declare HighSkyColor0_0=<0.3,0.3,0.7>;
#declare HighSkyColor1_0=<0.1,0.1,0.5>;
#declare HighSkyColor2_0=<0.05, 0.05, 0.3>;
#declare HighSkyColor3_0=<0.05, 0.05, 0.2>;
#declare LightColor_0=<1,0.3,0.2>;

#declare SimpleGround=0;
#declare BumpyGround=false;
#declare BumpyGroundScale=10;
#declare BumpyGroundHeight=0.2;
#declare SandTextureScale=0.5;
#declare UseSandHills=true;
#declare GroundContours=0;
// Camera(<position>,<look_at_target>)

////////////////////////////////

#declare CameraLocation=<0,0,1>;

#declare StandardCamera=camera { up <0,0,1> right <image_width/image_height,0,0> sky <0,0,1> location <0,0,0> look_at <0,1,0> }
//pov doc: direction_length = 0.5 * right_length / tan(angle / 2)
//         horiz.angle=2*atan( w/h / 2)                    = 83.3 (16:9)     67.4 (4:3)
//         diag.angle=2*atan( sqrt(w^2+h^2)/h /2)          = 91.1 (16:9)     79.6 (4:3)

#declare GlobalAmbient=2;

global_settings
 {
 max_trace_level 100
 adc_bailout 0.1
 ambient_light GlobalAmbient
// assumed_gamma 2.2
 assumed_gamma 1.0
 mm_per_unit 1
 subsurface { }
 }

#declare field_World_simtype=0;
#declare field_ODE_odeshape=0;

//////////////////////////////////////////
// colours
///////////////////////////////////////////
// stick:
#declare OldStickRGB = <0.65,0.65,0.55>;
#declare OldStick2RGB = <0.55,0.55,0.45>;
#declare StickRGB = <0.46,0.57,0.42>;
#declare StaminaRGB = <0.53,0.65,0.50>;
#declare IngestRGB = <0.93,0.93,0>;
#declare AsimilRGB = <0.14,0.9,0.14>;

/////////////////////
// muscle
#declare Muscle1RGB=<0.9,0.4,0.4>;
#declare Muscle2RGB=<0.9,0.2,0.2>;
#declare Muscle3RGB=<0.3,0.1,0.1>;

//{  distance 60   colour rgb<0.6, 0.8, 0.75> } //dist 16 ok

///////////////////////////////
#declare StickFinish = finish { phong 0.3 };

#declare SimpleStickTexture = texture
	{ pigment { color srgb Red }
	  finish { StickFinish }
   }

///////////////////////////////
// world:

#declare WORLD_FLAT=0;
#declare WORLD_BLOCKS=1;
#declare WORLD_SMOOTH=2;

#declare BORDER_NONE=0;
#declare BORDER_FENCE=1;
#declare BORDER_TELEPORT=2;

#declare WallTexture = texture { pigment {granite} }

#declare SandBumps = normal {granite 0.2 scale 0.1};
#declare SandBumpsX2 = normal {granite 0.4 scale 0.1};
#declare SandHills = normal {granite 0.2 scale 5};

#macro SandPigment()
  granite
  color_map
  {
    [ 0.0 srgb <0.90, 0.85, 0.65>*1.2 ]
    [ 0.7 srgb <0.80, 0.75, 0.55>*1.2 ]
  }
  scale SandTextureScale
#end

//#declare WetSandPigment = pigment { color White };

#macro WetSandPigmentAbove()
  granite
  color_map
  {
    [ 0.0 srgb <0.90, 0.85*1.05, 0.65*1.1> ]
    [ 0.7 srgb <0.80, 0.75*1.05, 0.55*1.1> ]
  }
  scale SandTextureScale
#end

#macro WetSandPigmentBelow()
  granite
  color_map
  {
    [ 0.0 srgb <0.90, 0.85*1.05, 0.75*1.05>*1.1 ]
    [ 0.7 srgb <0.80, 0.75*1.05, 0.65*1.05>*1.1 ]
  }
  scale SandTextureScale
#end

#macro Ground0Texture()
texture
	{ 
#if (GroundContours)
	pigment {
	 gradient z
		color_map
			{
			[ 0.0 srgb <1,0,0> ]
			[ 1.0 srgb <1,1,0> ]
			}
	 }

	}
  texture {
	pigment {
	 gradient z
		color_map
			{
			[ 0.0 srgbt <0,0,0,1> ]
			[ 0.8 srgbt <0,0,0,1> ]
			[ 0.8 srgbt <0,0,0,0.8> ]
			[ 1.0 srgbt <0,0,0,0.8> ]
			}
	 scale 0.1
	 }

#else	  
#if (SimpleGround)
    pigment {color White}
#else
//   {color rgb <1,0.9,0.6>}
#if(UseSandHills)
#local SandNormal=normal { average normal_map {
	  [ 1.0 SandBumpsX2 ]
	  [ 1.0 SandHills ] } }
#else
#local SandNormal= normal { SandBumps }
#end
#if (WetSandAboveWater>0)
    function { (z-(WetSandAboveWater+field_World_wrldwat+OFFSET_Z))/100+0.5+f_snoise3d(x*0.3,y*0.3,0)*0.001 }
    texture_map {
      [ 0.5-0.001 pigment { WetSandPigmentAbove() } ]
      [ 0.5+0.001 pigment { SandPigment() } normal { SandNormal } ]
    }
#else
    pigment { SandPigment() }
    normal { SandNormal }
#end //WetSand
#end //SimpleGround
#end
	}
#end

/////////////////////////////////

#declare UnderwaterTexture=texture
{
	pigment
    {
	 granite
	color_map
      { [ 0 color srgb Gray60 ]
		  [ 1 color srgb White ] }
	scale 2
    }
  finish {ambient 0.7/GlobalAmbient diffuse 0.3 }
}

/////////////////////////////////

#declare HighTexture=texture
{
	pigment
    {
	 crackle
	color_map
      { [ 0 color srgb Gray40 ]
		  [ 0.1 color srgb Gray80 ]
		  [ 1 color srgb White ] }
    }
}


///////////////////////////////////

/*
#declare HighTexture=texture
	{
	pigment
//   {color rgb <1,0.9,0.6>}
	 {
	 granite
		color_map
			{
			[ 0.0 rgb <0.95, 0.90, 0.60> ]
			[ 0.7 rgb <0.76, 0.70, 0.40> ]
			[ 1.0 rgb <0.76, 0.70, 0.40> ]
			}
	 scale .5  					//3
	 }
    normal {bumps 0.2 scale 0.02}
	}
*/

/////////////////////////////////////////

#declare WaterWaveSpeed = <0,0,0.2>;

#macro WaterTexture()
texture
{
#if (CameraIsUnderwater())
 pigment { bumps scale 2
 pigment_map {
            [0 rgbf <0.5,0.5,0.5,0.5>+UnderwaterSurfaceGlow*0.5 ]
            [0.5 rgbf UnderwaterSurfaceGlow ]
            [1 rgbf <0.5,0.5,0.5,0.5>+UnderwaterSurfaceGlow*0.5 ]
	    }
 }
#else
 pigment { rgbf<0.3, 0.7, 0.9, 0.9> }
#end
 finish { reflection { 0.5 fresnel on } conserve_energy
#if (CameraIsUnderwater())
   ambient 1.0/GlobalAmbient
#else
   ambient 0
#end
    }
#if (WaterTrueWaves=0)
 normal { bumps 0.5 scale 2 }
 translate WaterWaveSpeed*AnimFrame
#end
}
#end

#declare WaterIOR = 1.3;
#declare ForamIOR=WaterIOR;
#declare EnergyIOR = 1.0;

#macro WaterInterior()
interior { ior WaterIOR caustics 0.3 }
#end

///////////////////////////////////////////////
// LIGHT SOURCE

#macro StandardLight()
light_source
 {
  <0,-LightDistance,0>
  color LightColor
#if (LightSize>0)
  area_light  <LightSize,0,0> <0,0,LightSize>  8, 8 circular
  adaptive 1    jitter
#end
  rotate -x*LightAlt
  rotate z*LightAz
 }
#end

//////////////////////////////////////////////////////////
// SKY

#macro StandardSky()
sky_sphere {
    pigment {
        gradient y
        pigment_map {
            [0.01 srgb HorizonColor*SkyBrightness ] // horizon
            [0.25
    wrinkles
    turbulence 0.65
    octaves 6
    omega 0.7
    lambda 2
    color_map {
        [0.0 color srgb MidSkyColor0*SkyBrightness]
        [0.1 color srgb MidSkyColor1*SkyBrightness]
        [0.5 color srgb MidSkyColor2*SkyBrightness]
        [1.001 color srgb MidSkyColor3*SkyBrightness]
    }
    scale <1.5, 0.25, 1.5> rotate z*5]
  [0.60
    bozo
    color_map {
        [0.0 color srgb HighSkyColor0*SkyBrightness]
        [0.1 color srgb HighSkyColor1*SkyBrightness]
        [0.5 color srgb HighSkyColor2*SkyBrightness]
        [1.001 color srgb HighSkyColor3*SkyBrightness]
    }
    turbulence 0.65
    octaves 6
    omega 0.707
    lambda 2
    scale <1.5, 0.6, 1.5> rotate z*10]
        }
    }
 rotate <90,0,0>
}
#end

//background { color SkyBlue }

///////////////////////////////////////////////
// MANIPULATOR

#declare ManipulatorArm = union
 {
 cylinder { <0,0,0>,<0,0,10>,0.2 }
 sphere { <0,0,0>,0.3 }
 }
#declare ManipulatorFinger = union
 {
 cylinder { <0,0,0>,<-1,0,-1>,0.17 }
 sphere { <-1,0,-1>,0.17 }
 cone { <-1,0,-1>,0.17,<0,0,-2>,0.1 }
 sphere { <0,0,-2>,0.1 }
 }
#declare ManipulatorTexture=Aluminum

#macro Manipulator(Location,State)
union
 {
 object {ManipulatorArm}
 object {ManipulatorFinger rotate <0,5+25*abs(State),0>}
 object {ManipulatorFinger rotate <0,5+25*abs(State),180>}
 translate <0,0,2>
 texture {ManipulatorTexture}
 scale 0.5
 rotate <0,0,180>*abs(State)
 translate Location+<0,0,10>*abs(State)
 }
#end

///////////////////////////////////////////////
// STONES

  #declare Stone =
    sphere { <0,0,0>,1
    scale <1.5,0.8,1>
    texture {T_Stone25 scale 2 }
    normal { crackle 3.0 scale 2 }
    }
    
  #macro LineOfStones(Begin,End,Scale,Density,Noise)
    #local Len=vlength(Begin-End);
    #local Count=Density*Len;
    #local R=seed(321);
    #local Pos=Begin;
    #local Step=(End-Begin)/Count;
    #while(Count>0)
       object { Stone 
	 scale Scale
	 rotate <360*rand(R),360*rand(R),360*rand(R)>
	 translate Pos + <rand(R)-0.5,rand(R)-0.5,rand(R)-0.5>*Noise
         }
       #local Count=Count-1;
       #local Pos=Pos+Step;
    #end
  #end

  #macro ArcOfStones(Center,Radius,Begin,End,Scale,Density,Noise)
    #local Len=2*pi*Radius*(End-Begin)/360;
    #local Count=Density*Len;
    #local R=seed(432);
    #local A=Begin;
    #local Step=(End-Begin)/Count;
    #while(Count>0)
       object { Stone 
	 scale Scale
	 rotate <360*rand(R),360*rand(R),360*rand(R)>
	 translate Center + <cos(radians(A)),sin(radians(A)),0>*Radius + <rand(R)-0.5,rand(R)-0.5,rand(R)-0.5>*Noise
         }
       #local Count=Count-1;
       #local A=A+Step;
    #end
  #end

  #macro UnderWaterStones()
    #local Scale=0.5;
    #local Density=20;
    #local Noise=2;
    #local Margin=(Scale*1.5+Noise)/2;
    LineOfStones(<0,field_World_wrldsiz+Margin,0>,<field_World_wrldsiz,field_World_wrldsiz+Margin,0>,Scale,Density,Noise)
    LineOfStones(<0,-Margin,0>,<field_World_wrldsiz,-Margin,0>,Scale,Density,Noise)
    LineOfStones(<field_World_wrldsiz+Margin,0,0>,<field_World_wrldsiz+Margin,field_World_wrldsiz,0>,Scale,Density,Noise)
    LineOfStones(<-Margin,0,0>,<-Margin,field_World_wrldsiz,0>,Scale,Density,Noise)
    #if (Margin>0)
      ArcOfStones(<field_World_wrldsiz,field_World_wrldsiz,0>,Margin,0,90,Scale,Density,Noise)
      ArcOfStones(<0,0,0>,Margin,180,270,Scale,Density,Noise)
      ArcOfStones(<field_World_wrldsiz,0,0>,Margin,270,360,Scale,Density,Noise)
      ArcOfStones(<0,field_World_wrldsiz,0>,Margin,90,180,Scale,Density,Noise)
    #end
  #end      


///////////////////////////////////////////////
// FENCE

#declare FenceSegmentLen = 5;
#declare FenceHeight = 4;
#declare FenceBeamSpacing = 1;
#declare FenceBeamHeight = 1;
#declare FenceBeamsNum = 3;

#declare FenceTexture = texture { T_Wood17 scale 0.6 }

// FENCE

#declare FenceBeam = object
{
cylinder { <-0.4-FenceSegmentLen/2,0,0>,<FenceSegmentLen/2+0.4,0,0>,0.15
 texture {FenceTexture rotate <0,90,0> }}
}
#declare FencePost = union
{
box { <-0.2,-0.2,0>,<0.2,0.2,FenceHeight> }
  union
  {
  triangle { <-0.2,-0.2,0>,<-0.2,0.2,0>,<0,0,0.2> }
  triangle { <-0.2, 0.2,0>,< 0.2,0.2,0>,<0,0,0.2> }
  triangle { < 0.2, 0.2,0>,< 0.2,-0.2,0>,<0,0,0.2> }
  triangle { < 0.2,-0.2,0>,<-0.2,-0.2,0>,<0,0,0.2> }
   translate FenceHeight*z
  }
texture { FenceTexture }
}

#declare FenceSegment = union
{
object { FencePost }
#local N=FenceBeamsNum;
#local Y=FenceHeight;
#local R=seed(3333);
#while (N>0)
 #local Y=Y-FenceBeamSpacing;
 #local N=N-1;
 object { FenceBeam rotate <360*rand(R),(rand(R)-0.5)*20*atan2(FenceBeamSpacing,FenceSegmentLen),0> translate <FenceSegmentLen/2,0,Y> }
#end
}

#macro Fence()
#local K=field_World_wrldsiz;
#while (K>7) #local K=K/2; #end
#local K=K+0.0001;
#local X=0;
#while (X<field_World_wrldsiz-0.001)
object {FenceSegment scale <K/FenceSegmentLen,1,1>
	 translate <X,0,OFFSET_Z> }
object {FenceSegment scale <K/FenceSegmentLen,1,1>
    rotate <0,0,90>
	 translate <field_World_wrldsiz,X,OFFSET_Z> }
object {FenceSegment scale <K/FenceSegmentLen,1,1>
    rotate <0,0,180>
	 translate <field_World_wrldsiz-X,field_World_wrldsiz,OFFSET_Z> }
object {FenceSegment scale <K/FenceSegmentLen,1,1>
    rotate <0,0,270>
	 translate <0,field_World_wrldsiz-X,OFFSET_Z> }
#local X=X+K;
#end
#end

#macro TeleportBoundary()
difference 
 {box {<-5,-5,0.001>,<field_World_wrldsiz+5,field_World_wrldsiz+5,1>}
  box {<0,0,0>,<field_World_wrldsiz,field_World_wrldsiz,1.001>}

 pigment {color rgbf <1,1,1,1>}
 interior {media
	{ emission 0.1
	  density {gradient z translate <0,0,2> }
	}
   }
 hollow
 scale <1,1,7>
 translate <0,0,OFFSET_Z>
 }
#end

/////////////////////////////////////
// ENERGY

#declare EnergyTexture = texture
{
pigment { marble
  color_map {
   [ 0 color srgbf <1,1,1,0.9> ]
   [ 0.1 color srgbf <1,0.95,0.85,0.9> ]
   [ 0.2 color srgbf <1,0.95,0.8,0.95> ]
   [ 0.6 color srgbf <1,0.9,0.8,0.95>]
            }
		turbulence 0.5
        }
	finish { ambient 1.0/GlobalAmbient
	         phong 0.4 }
}

#macro BeginEnergy()
#end

#macro EndEnergy()
#if (NumEnergyComponents>0)
blob {
 threshold 1
 #local i=0;
 #while (i<NumEnergyComponents)
  #if (EnergyComponents[i].z > (field_World_wrldwat+OFFSET_Z))
  component 2.0 EnergyComponentsPower[i] EnergyComponents[i] 
  #end
  #local i=i+1;
 #end
 texture {EnergyTexture translate <AnimFrame*0.03,0,0>} hollow
 }
blob {
 threshold 1
 #local i=0;
 #while (i<NumEnergyComponents)
  #if (EnergyComponents[i].z <= (field_World_wrldwat+OFFSET_Z))
  component 2.0 EnergyComponentsPower[i] EnergyComponents[i] 
  #end
  #local i=i+1;
 #end
 texture {EnergyTexture translate <AnimFrame*0.03,0,0>} hollow interior { ior WaterIOR }
 }
#end
#end

#declare REnergySeed=seed(300+mod(AnimFrame,20));

#macro ParametrizedPureEnergy(SphereAmbientColor,StarAmbientColor)
union
{
 sphere
  { 0, 1 pigment { bozo color_map {[0.0 color srgbt <1,1,1,0.95>][1.0 color srgbt <1,1,1,1>]}
                 scale 0.2+0.2*rand(REnergySeed) turbulence 2 }
          finish { ambient srgb SphereAmbientColor/GlobalAmbient } hollow no_shadow
  } 
#local N=10;
#while (N>0)
 sphere { 0, 0.7 pigment {color srgbt <1,1,1,0.85>}
         finish { ambient srgb StarAmbientColor/GlobalAmbient } hollow no_shadow scale <1,0.1,0.1>
 rotate <360.0*rand(REnergySeed),360.0*rand(REnergySeed),360.0*rand(REnergySeed)>
  }
#local N=N-1;
#end
}
#end
  
#declare PureEnergy=object { ParametrizedPureEnergy(5*<2,1,0.5>,5*<1,1,0.5>) };

#macro InnerEnergyObject(En)
light_source
{
  0 color <1,0.5,0.2>*En
#if (!EnergyLightShadows)
  shadowless
#end
fade_distance 1 fade_power 2 looks_like { object { PureEnergy scale 0.5*En } }
}
#end

/*
#declare PureEnergy=
sphere { 0, 1
  pigment { rgbt<1, 1, 1, 1> }
  interior {
    media {
      emission <5,5,2>
      density {
        spherical
        turbulence 0.5
//        color_map {
//          [0.0 color rgb <0, 0, 0>]
//          [0.4 color rgb <1, 0, 0>]
//          [0.7 color rgb <1, 1, 0.5>]
//          [1.0 color rgb <1, 1, 1>]
//        } 
      }
    }
  }
  hollow
}
*/


#declare MAX_ENERGY_COMPONENTS=5000;
#declare EnergyComponents=array[MAX_ENERGY_COMPONENTS]
#declare EnergyComponentsPower=array[MAX_ENERGY_COMPONENTS]
#declare NumEnergyComponents=0;

#macro AddEnergyComponent(Location,Power)
#if (NumEnergyComponents<MAX_ENERGY_COMPONENTS)
#declare EnergyComponents[NumEnergyComponents]=Location;
#declare EnergyComponentsPower[NumEnergyComponents]=Power;
#declare NumEnergyComponents=NumEnergyComponents+1;
#end
#end

#declare EnergyParticleScale=0.15;

#macro EnergyParticle(Id,TTL,Location,Amount)
  #local S=0.75+rand(seed(Id))*0.5;
  Energy(Location,pow(Amount,1/3)*EnergyParticleScale*S,true)
#end

#declare EnergyTransferCounter=0;
#declare SkipEnergyTransfer=10;

#macro EnergyTransferObject(En)
#declare EnergyTransferCounter=EnergyTransferCounter+1;
#if (EnergyTransferCounter >= SkipEnergyTransfer)
  #declare EnergyTransferCounter=0;
  
light_source
{
  0 color <1,0.9,0.8>*En*0.5*SkipEnergyTransfer
#if (!EnergyLightShadows)
  shadowless
#end
fade_distance 1 fade_power 2 looks_like { object { ParametrizedPureEnergy(3*<2,1.8,1.0>,3*<1,1,0.9>) } scale 0.5*En }
}
#else
  
  object { ParametrizedPureEnergy(3*<2,1.8,1.0>,3*<1,1,0.9>) scale 0.5*En }
  
#end
#end

#macro Energy(Location,Amount,IsTransfer)
#local R=max(EnergyRadiusMin,min(EnergyRadiusMax,Amount));
#if (R>0.0001)
#if(IsTransfer)
 AddEnergyComponent(Location,3*R)
// object { InnerEnergyObject(R) translate Location }
 object { EnergyTransferObject(R) translate Location }
#else
 AddEnergyComponent(Location,2*R)
 object { InnerEnergyObject(R) translate Location + <0,0,0.1> }
#end
#end
#end

/////////////////////////////////////

#declare StickEndR=0.15;

#macro CreatureStickEnd(Location,Rotation)
union {
#if (field_World_simtype & field_ODE_odeshape)
 box { <-0.2,-0.2,-0.2>,<0.2,0.2,0.2>
#else
sphere { <0,0,0>,StickEndR
#end
 texture
   {
   pigment { StickBasePigment }
   normal { bumps 0.5*field_p_fr scale 0.1 }
//   finish { phong (1-field_p_fr) }
//normal { StickRoughnessNormal }
   finish { StickFinish }  
   scale 0.2
   }
}       
#if (St_Recept_S)

#local N=25;
union {
#while (N>0)
  difference { cylinder { <0,0.1,0>,<0,0.2,0>,0.02} cylinder { <0,0.13,0>,<0,0.21,0>,0.018} pigment { color srgb <1,0.6,0> } rotate<0,0,rand(CreatureUniqueSeed)*60-30>  rotate<rand(CreatureUniqueSeed)*360,0,0> }
  
  #local N=N-1;
#end
  translate <-0.2,0,0>
}
  
#end // Recept_S

#if (St_Recept_T)
 #local N=3;
 #while (N>0)
  #local N=N-1;
  union {
    cylinder { <0,0.1,0>,<0.15,0.12,0>,0.01 }
    sphere { <0.15,0.12,0>,0.01 }
    cone { <0.15,0.12,0>,0.01,<0.3,0.15,0>,0.007 }
    sphere { <0.3,0.15,0>,0.02 }
    texture { pigment { color srgb <0.1,0.1,0.1> } 
      finish { phong 1 } }
    rotate <45+120*N,0,0>
    }
 #end
#end // St_Recept_T

#if (field_p_ing > 0.05)
 #local N=field_p_ing*50;
 #local R=seed(123);
 #while (N>0)
  #local N=N-1;
  sphere { <0,0,StickEndR-0.01> 0.03
  texture { 
//	pigment { color srgb IngestRGB }
        pigment  { spiral1 10 color_map { [0 color srgb IngestRGB ] [1 color srgb IngestRGB*0.9] } }
        normal { spiral1 10 bump_size 0.1 }
	finish { phong 1 }
	}
   rotate <0,360*rand(R),360*rand(R)>
    }
 #end
#end
rotate Rotation*57.3
translate Location
}
#end

/////////////////////////////////////////////////////////////

#declare MuscleTexture = texture
{
pigment
 {
 gradient x   frequency 1   turbulence 0.1   ramp_wave
 color_map {[0 color srgb Muscle1RGB]  [0.2 color srgb Muscle2RGB]  [1 color srgb Muscle3RGB]}
 }
normal { radial 0.05  scallop_wave   frequency 15   rotate <0,0,90> }
finish { phong 1 }
scale <0.2,1,1>
}

#declare Muscle = union
{
sphere {<0.2,0,0>,0.1}
cylinder {<0,0,0>,<0.2,0,0>,0.1}
sphere {<0,0,0>,0.1}
texture {MuscleTexture}
}

#macro ShapeTexture()
   texture {
     pigment {
             ripples scale 0.2 turbulence 1.5
             color_map {
	               [0.0 color srgb <field_p_vr, field_p_vg, field_p_vb>*0.8]
	               [1.0 color srgb <field_p_vr, field_p_vg, field_p_vb>*1.1]
		       }
              scale <1.0/field_p_sx,1.0/field_p_sy,1.0/field_p_sz>
              }
     finish { phong 1 }
     }
#end


#macro CreatureUIDAsNumber()
 val(substr(field_Creature_uid,2,strlen(field_Creature_uid)-1))
#end

#declare RetiFunction = function(x,y,z)
  {
    min(1,abs(1-mod(100*(pi+atan2(x+f_noise3d(10*x,y,10*z)*sqrt(x*x+z*z)*0.2,z+f_noise3d(10*x,y+10,10*z)*sqrt(x*x+z*z)*0.2))/(2*pi),2))*sqrt(x*x+z*z)+max(0,sqrt(x*x+z*z)-0.95))
  }

#declare RetiColorMix = <0.45, 0.64, 0.45>;
  
#macro ReticulopodiaShape(Location,Rotation)
  #if(1)
#if (field_p_sy>0.1) //hide the first "technical" reticulopodia
#local Individual=AnimFrame*0.1+CreatureUIDAsNumber();
#local RGB=<field_p_vr,field_p_vg,field_p_vb>;
#local RGBMore=RGB+(RGB-<0.75,0.75,0.75>)*6;
#local RGB=<clip(RGBMore.x,0,1),clip(RGBMore.y,0,1),clip(RGBMore.z,0,1)>;
isosurface {
  function { sqrt(pow(RetiFunction(x,Individual,z),2)+y*y*4)-0.05 }
  contained_by { box { <-1,-0.05,-1>,<1,0.05,1> } }
  accuracy 0.001
  max_gradient 50
  rotate <0,0,90>
  scale <1,field_p_sy,field_p_sz>
  rotate Rotation*57.3
  translate Location
  texture {
     pigment { color rgb RGB }
//     finish { phong 0.8 ambient 0.3 diffuse 0.7 }
     finish { phong 0.8 ambient 0.2/GlobalAmbient }
     }
  normal {bumps 0.2 scale 0.005}
  }
#end //if (field_p_sy>0.1)
  #end
#end
  
#declare field_Creature_energy0=1; //exported in newer pov files, declared here so the formula does not break on old files

#macro NutrientShape(Location,Rotation)
#local EnergyScaling=field_Creature_energy/field_Creature_energy0;
#local WaveTime=AnimFrame/100;
#local WaveAmount=1.0;
#local Individual=sin(CreatureUIDAsNumber());
isosurface{
  function{  sqrt( x*x + y*y+ z*z)
              - (0.5 + f_noise3d( x*2 , y*2 , z*2 + 5*Individual )*1.90)*EnergyScaling
          }
  accuracy 0.002
  max_gradient 5
  contained_by { box { -2.5, 2.5 } }

  texture{ pigment{ color rgb<0.1, 0.4, 0.2>} normal { bumps 0.9 scale 0.035 } }
  scale 0.5
  rotate Rotation*57.3
  translate Location+<0,0,-1>
} //isosurface
#end
  
#macro CylinderShape(Location,Rotation)
cylinder
  {
  <-1,0,0>,<1,0,0>,1
  ShapeTexture()
  scale <field_p_sx,field_p_sy,field_p_sz>
  rotate Rotation*57.3
  translate Location
  }
#end

#macro CuboidShape(Location,Rotation)
box
  {
  <-1,-1,-1>,<1,1,1>
  ShapeTexture()
  scale <field_p_sx,field_p_sy,field_p_sz>
  rotate Rotation*57.3
  translate Location
  }
#end

#macro EllipsoidShape(Location,Rotation)
sphere
  {
  <0,0,0>,1
  ShapeTexture()
  scale <field_p_sx,field_p_sy,field_p_sz>
  rotate Rotation*57.3
  translate Location
  }
#end

#declare StickRoughnessNormal = normal { bumps 0.2 scale 0.04 turbulence 0.3 };
#declare StickCrackleNormal = normal { pigment_pattern { crackle color_map {
    [0.0, rgb 0.0] [0.3, rgb 0.1] [0.8, rgb 0.9] [1.0, rgb 1.0] } } 1.2 };
#declare StickBasePigment=pigment { crackle color_map{ [0.45 color srgb StickRGB] [0.55 color srgb StickRGB*0.87] } scale 0.2 };
  
#declare StickBaseTexture=texture
{
pigment { StickBasePigment }
normal { StickRoughnessNormal }
finish { StickFinish }
}

//#declare StickBaseTextureCombinedNormals=texture { StickBaseTexture normal { average normal_map { [ 1.0 StickRoughnessNormal ] [ 0.01 StickCrackleNormal ] } } };
//#declare StickBaseTextureCombinedNormals=texture { StickBaseTexture normal { StickCrackleNormal bumps 2 } };
  
#declare CreatureStickSegments=3;
#declare CreatureStickSegmentsPos=array[CreatureStickSegments+3];
#declare CreatureStickSegmentsR=array[CreatureStickSegments+3];

#declare GreenCellsFunction=function { min(1,max(0,f_crackle(x,y,z)+f_snoise3d(x*20,y*20,z*20)*0.2)) };

#declare StickGreenCellTexture=texture
{
pigment { color srgb AsimilRGB }
normal { StickCrackleNormal }
finish { StickFinish }
}

#macro RotMuscle()
union
  {
#local R=0.15;
#local NumSeg=18;
#local Len=R*(2*pi)/NumSeg;
#local I=0;
#local Randomness=0.01;
#while(I<NumSeg)
  #local A=I*360/NumSeg;
  #local P=<0,R*cos(radians(A)),R*sin(radians(A))>;

  object
    { Muscle
    scale <Len*5,St_Speed,St_Speed>
    rotate <0,90,0>
    rotate <A,0,0>
    translate P + <(rand(CreatureUniqueSeed)-0.5)*Randomness,(rand(CreatureUniqueSeed)-0.5)*Randomness,(rand(CreatureUniqueSeed)-0.5)*Randomness>
    }
  #local I=I+1;
#end
  }
#end //RotMuscle
  
#macro CreatureStick(Location1,Location2,Rotation)

#local NominalStickLength=vlength(Location1-Location2);
  
#local StickTexture=texture
{
  function { GreenCellsFunction(x,y,z) }
  texture_map {
             [pow(0.85-St_Asimil,1.8) StickBaseTexture]
             [pow(0.95-St_Asimil,1.8) StickGreenCellTexture] }
//normal { StickCrackleNormal }
translate <0,rand(CreatureUniqueSeed)*10,rand(CreatureUniqueSeed)*10>
//finish { StickFinish }
scale 0.2
}

#if (CreatureStickSegments>1)
#local I=0;
#local DP=<NominalStickLength,0,0>/CreatureStickSegments;
#local P=<0,0,0>-DP;
#local RP=0.02;
#local R=0.1;
#local RR=0.01;
#while (I<(CreatureStickSegments+3))
  #declare CreatureStickSegmentsPos[I]=P+(<rand(CreatureUniqueSeed),rand(CreatureUniqueSeed),rand(CreatureUniqueSeed)>-<0.5,0.5,0.5>)*RP;
  #declare CreatureStickSegmentsR[I]=R+(rand(CreatureUniqueSeed)-0.5)*RR;
  #local I=I+1;
  #local P=P+DP;
#end
#end

union
{
#if (St_Recept_G)
union { difference {
#end
#if (field_World_simtype & field_ODE_odeshape)
 box { <0,-0.1,-0.1>,<NominalStickLength,0.1,0.1> /*texture { StickBaseTexture } texture { StickGreenCellsTexture }*/ texture {StickTexture} }
#else
#if (CreatureStickSegments>1)
  #local StickCenterP=(CreatureStickSegmentsPos[floor((CreatureStickSegments+3)/2)]+CreatureStickSegmentsPos[floor((CreatureStickSegments+2)/2)])/2;
  sphere_sweep
  {
    cubic_spline CreatureStickSegments+3
    #local I=0;
    #while(I<(CreatureStickSegments+3))
      CreatureStickSegmentsPos[I],CreatureStickSegmentsR[I]
      #local I=I+1;
    #end
    texture { StickTexture }
  }
#else
 #local StickCenterP=<NominalStickLength/2,0,0>;
 cylinder { <0,0,0>,<NominalStickLength,0,0>,0.1 texture {StickTexture} }
#end
#end
#if (St_Recept_G)
  cylinder { <-0.12,0,0>,<0.12,0,0>,0.12 translate StickCenterP } //subtract this cylinder from stick
  }
 cylinder { <-0.121,0,0>,<0.121,0,0>,0.11
  texture
    {
    pigment  {color srgbf <1.0, 1.0, 1.0, 0.9>}
    finish   {ambient 0   diffuse 0   reflection 0.5  phong 0.3   phong_size 60}
    }
   interior {ior 1.4}
   translate StickCenterP
          }
 sphere { <0,0,0>,0.03
          texture { pigment { color srgb <1,0,0> } finish { phong 1 } }
          translate StickCenterP
        }
 }
#end // Recept_G

#if (field_j_stam>0.07)
 #local N=6;
 #while (N>0)
  #local N=N-1;
   
#if (CreatureStickSegments>1)
  #local A=N*60;
  sphere_sweep
  {
    cubic_spline CreatureStickSegments+3
    #local I=0;
    #while(I<(CreatureStickSegments+3))
      <0,CreatureStickSegmentsR[I]*cos(radians(A)),CreatureStickSegmentsR[I]*sin(radians(A))>+CreatureStickSegmentsPos[I],field_j_stam*0.04
      #local I=I+1;
    #end
    texture { StaminaTexture }
  }
#else

 cylinder { <0,0,0.1>,<1,0,0.1>,field_j_stam*0.04 rotate <N*60,0,0> texture { StaminaTexture } }

#end
   
 #end
#end
scale <vlength(Location2-Location1)/NominalStickLength,1,1>
rotate Rotation*57.3
translate Location1
}
#end

#declare StaminaTexture = texture { pigment { color srgb StaminaRGB } normal { bumps 0.5 scale 0.015 } };
 
////////////////////////////////////////////////

#macro GetGroundTexture(Height)
#if (Height>field_World_wrldwat)
#if (Height>(field_World_wrldwat+2))
texture {HighTexture}
#else
Ground0Texture()
#end
#else
texture {UnderwaterTexture}
#end
#end

#macro CommonWorldObjects()
#end

#declare field_World_wrldtyp=0;
#declare field_World_wrldwat=0;
#declare field_World_wrldbnd=0;

#declare BumpyHeightfield = height_field
        {
        png "bumps-wrap.png" gamma 1.0
	smooth
	rotate <90,0,0> translate <0,1,0>
        }
#declare BumpyHeightfieldResolution=1026; //1026x1026 image, the last two rows and columns copied from the first 2 rows/columns
#declare BumpyHeightfieldOverlap=(BumpyHeightfieldResolution-1)/(BumpyHeightfieldResolution-2);
//#declare BumpyHeightfieldOverlap=33.0/32.0;

#macro BeginWorld()
#ifdef (Override_OFFSET_Z)
 #declare OFFSET_Z=Override_OFFSET_Z;
#else
 #if (field_World_simtype=1)
  #declare OFFSET_Z=0.0;
 #else
  #declare OFFSET_Z=-0.15;
 #end
#end
CommonWorldObjects()
#switch (field_World_wrldtyp)
#case (WORLD_FLAT)
  
#if(!BumpyGround)
plane { z,OFFSET_Z GetGroundTexture(0) }
#else
#local N=floor(field_World_wrldsiz*2/BumpyGroundScale);
#local BumpySize=N*BumpyGroundScale;
#local BumpyXYMin=0.5*(field_World_wrldsiz-BumpySize);
#local BumpyXYMax=0.5*(field_World_wrldsiz+BumpySize);
#local BumpyTileScaling=BumpySize/N*BumpyHeightfieldOverlap;
#local CutBox = box { <BumpyXYMin ,BumpyXYMin,OFFSET_Z-2*BumpyGroundHeight>, <BumpyXYMax,BumpyXYMax,OFFSET_Z+BumpyGroundHeight> };
union
  {
  difference
     {
     plane { z,OFFSET_Z GetGroundTexture(0) }
     box { CutBox }
     GetGroundTexture(0)
     }
//  #local N=1;
  #local X=0;
//  #local R=seed(6);
  #while(X<N)
    #local Y=0;
    #while(Y<N)
//      #if (rand(R)>0.4)
      height_field
        {
	BumpyHeightfield
        scale <BumpyTileScaling,BumpyTileScaling,BumpyGroundHeight>
        translate <BumpyGroundScale*X+BumpyXYMin,BumpyGroundScale*Y+BumpyXYMin,OFFSET_Z>  
//	pigment { color White }
        GetGroundTexture(0)
        }
//      #end
      #local Y=Y+1;
    #end
    #local X=X+1;
  #end
/* (does not cooperate with media)
  isosurface
    {
    function { z-OFFSET_Z + f_noise3d(GroundNoiseScale*x,GroundNoiseScale*y,GroundNoiseScale*z)*GroundNoiseHeight }
//    contained_by { box { CutBox } }
    contained_by { box { <-field_World_wrldsiz,-field_World_wrldsiz,-0.5+OFFSET_Z>,
           <2*field_World_wrldsiz,2*field_World_wrldsiz,0.5+OFFSET_Z> } }
    GetGroundTexture(0)
    }
*/
  }
#end
  
#break
#end
#end //#macro BeginWorld()

#macro FlatBlock(X,Y,Z)
box { <X,Y,-51>,<X+WorldBlockX,Y+WorldBlockY,Z+OFFSET_Z>
	GetGroundTexture(Z)}
#end

#macro SlideBlockX(X,Y,Z,Z2)
#local X2=X+WorldBlockX;
#local Y2=Y+WorldBlockY;
#local ZO=Z+OFFSET_Z;
#local ZO2=Z2+OFFSET_Z;
union {
 polygon {5,<X,Y,ZO>,<X2,Y,ZO2>,<X2,Y2,ZO2>,<X,Y2,ZO>,<X,Y,ZO>}
 polygon {5,<X,Y,ZO>,<X,Y2,ZO>,<X,Y2,-10>,<X,Y,-10>,<X,Y,ZO>}
 polygon {5,<X2,Y,ZO2>,<X2,Y2,ZO2>,<X2,Y2,-10>,<X2,Y,-10>,<X2,Y,ZO2>}
 polygon {5,<X,Y2,ZO>,<X2,Y2,ZO2>,<X2,Y2,-10>,<X,Y2,-10>,<X,Y2,ZO>}
 polygon {5,<X,Y,ZO>,<X2,Y,ZO2>,<X2,Y,-10>,<X,Y,-10>,<X,Y,ZO>}
  GetGroundTexture(((Z+Z2)/2)) }
#end

#macro SlideBlockY(X,Y,Z,Z2)
#local X2=X+WorldBlockX;
#local Y2=Y+WorldBlockY;
#local ZO=Z+OFFSET_Z;
#local ZO2=Z2+OFFSET_Z;
union {
 polygon {5,<X,Y,ZO>,<X2,Y,ZO>,<X2,Y2,ZO2>,<X,Y2,ZO2>,<X,Y,ZO>}
 polygon {5,<X,Y,ZO>,<X,Y2,ZO2>,<X,Y2,-10>,<X,Y,-10>,<X,Y,ZO>}
 polygon {5,<X2,Y,ZO>,<X2,Y2,ZO2>,<X2,Y2,-10>,<X2,Y,-10>,<X2,Y,ZO>}
 polygon {5,<X,Y2,ZO2>,<X2,Y2,ZO2>,<X2,Y2,-10>,<X,Y2,-10>,<X,Y2,ZO2>}
 polygon {5,<X,Y,ZO>,<X2,Y,ZO>,<X2,Y,-10>,<X,Y,-10>,<X,Y,ZO>}
  GetGroundTexture(((Z+Z2)/2)) }
#end

#macro SmoothMapNormal(i,j)
#if (i>0)
#local zly=MapData[i-1][j];
#else
#local zly=MapData[i][j];
#end
#if (j>0)
#local zlx=MapData[i][j-1];
#else
#local zlx=MapData[i][j];
#end
#if (i<ny)
#local zry=MapData[i+1][j];
#else
#local zry=MapData[i][j];
#end
#if (j<nx)
#local zrx=MapData[i][j+1];
#else
#local zrx=MapData[i][j];
#end
#local N=<(zlx-zrx)/dx,(zly-zry)/dy,2>;
vnormalize(N)
#end

/* stare!
#macro SmoothMapNormal(i,j)
#if (i>0)
#local zlx=MapData[i-1][j];
#else
#local zlx=MapData[i][j];
#end
#if (j>0)
#local zly=MapData[i][j-1];
#else
#local zly=MapData[i][j];
#end
#if (i<ny)
#local zrx=MapData[i+1][j];
#else
#local zrx=MapData[i][j];
#end
#if (j<nx)
#local zry=MapData[i][j+1];
#else
#local zry=MapData[i][j];
#end
<(zlx-zrx)/dy,(zly-zry)/dx,2>
#end
*/

#macro InterpolateMap(X,Y)

  #if (X<0)
    #local X=0;
  #end
  #if (Y<0)
    #local Y=0;
  #end
  #local Xi=floor(X);
  #local Xi1=Xi+1;
  #local Yi=floor(Y);
  #local Yi1=Yi+1;
  #if (X>=(dimension_size(MapData,2)-1))
    #local Xi=dimension_size(MapData,2)-1;
    #local Xi1=Xi;
    #local X=Xi;
  #end
  #if (Y>=(dimension_size(MapData,1)-1))
    #local Yi=dimension_size(MapData,1)-1;
    #local Yi1=Yi;
    #local Y=Yi;
  #end
  
  #local Z=MapData[Yi][Xi];
  #local Zx=MapData[Yi][Xi1];
  #local Zy=MapData[Yi1][Xi];
  #local Zxy=MapData[Yi1][Xi1];
  
  #local izx=Z+(Zx-Z)*(X-Xi);
  #local izy=Zy+(Zxy-Zy)*(X-Xi);
  izx+(izy-izx)*(Y-Yi)
  
#end
  
#ifndef (MakeSmoothWorld)
#macro MakeSmoothWorld(WithInfinitePlane)
union
  {
#local i=0;
#local j=0;
#local X=0;
#local Y=0;
#local nx=dimension_size(MapData,2)-1;
#local ny=dimension_size(MapData,1)-1;
#local dx=field_World_wrldsiz/nx;
#local dy=field_World_wrldsiz/ny;

#if (HeightfieldMode<2)
mesh {
#end
  
#while (i<nx)
  #declare j=0;
  #declare Y=0;
  #while (j<ny)
    
#local Z=MapData[j][i];
#local Zx=MapData[j][i+1];
#local Zy=MapData[j+1][i];
#local Zxy=MapData[j+1][i+1];
    
#if (HeightfieldMode=0)
triangle { <X,Y,Z>,<X+dx,Y,Zx>,<X+dx,Y+dy,Zxy> }
triangle { <X,Y,Z>,<X+dx,Y+dy,Zxy>,<X,Y+dy,Zy> }
#end

#if (HeightfieldMode=1)
smooth_triangle { <X,Y,Z>,SmoothMapNormal(j,i),
		  <X+dx,Y,Zx>,SmoothMapNormal(j,i+1),
		  <X+dx,Y+dy,Zxy>,SmoothMapNormal(j+1,i+1) }
smooth_triangle { <X,Y,Z>,SmoothMapNormal(j,i),
		  <X+dx,Y+dy,Zxy>,SmoothMapNormal(j+1,i+1),
		  <X,Y+dy,Zy>,SmoothMapNormal(j+1,i) }
#end
    
#if (HeightfieldMode=2)

#local X1=X+dx/3;
#local X2=X+2*dx/3;
#local X3=X+dx;
#local Y1=Y+dy/3;
#local Y2=Y+2*dy/3;
#local Y3=Y+dy;
    
#if (i>0)
#local im=i-1;
#else
#local im=i;
#end

#if (i<(nx-1))
#local ip=i+2;
#else
#local ip=i+1;
#end

#if (j>0)
#local jm=j-1;
#else
#local jm=j;
#end

#if (j<(ny-1))
#local jp=j+2;
#else
#local jp=j+1;
#end
    
#local sx0=(Zx-MapData[j][im])/6;
#local sy0=(Zy-MapData[jm][i])/6;
#local sx1=(MapData[j][ip]-Z)/6;
#local sy1=(Zxy-MapData[jm][i+1])/6;
#local sx2=(Zxy-MapData[j+1][im])/6;
#local sy2=(MapData[jp][i]-Z)/6;
#local sx3=(MapData[j+1][ip]-Zy)/6;
#local sy3=(MapData[jp][i+1]-Zx)/6;

#end  //HeightfieldMode=2

#if ((HeightfieldMode=2) | (HeightfieldMode=3))
  bicubic_patch
    {
      type 1 u_steps 4 v_steps 4 /*flatness 0.005*/ /*flatness 0.02*/ // may cause "cracking" http://wiki.povray.org/content/Reference:Bicubic_Patch
#end
      
#if (HeightfieldMode=3)
      
#local ii=0;
#while(ii<=3)
  #local jj=0;
  #while(jj<=3)

    #local mode=0;
    #local px=X+dx/3*ii;
    #local py=Y+dy/3*jj;

    #local xm=i+ii/3;
    #local ym=j+jj/3;
    
    #if (((ii=0) & (i>0)) | ((ii=3) & (i<(nx-1))))
      #local mp1=<xm+1/3,ym>;
      #local mp2=<xm-1/3,ym>;
      #local mode=mode+1;
    #end

    #if (((jj=0) & (j>0)) | ((jj=3) & (j<(ny-1))))
      #local mp1=<xm,ym+1/3>;
      #local mp2=<xm,ym-1/3>;
      #local mode=mode+1;
    #end
    
    #switch(mode)
      
      #case(0) //middle
        #local pz=InterpolateMap(xm,ym);
      #break

      #case(1) //edge
        #local pz_1=InterpolateMap(mp1.x,mp1.y);
        #local pz_2=InterpolateMap(mp2.x,mp2.y);
        #local pz=(pz_1+pz_2)/2;
      #break
      
      #case(2) //node
        #local pz_1=InterpolateMap(xm-1/3,ym-1/3);
        #local pz_2=InterpolateMap(xm-1/3,ym+1/3);
        #local pz_3=InterpolateMap(xm+1/3,ym-1/3);
        #local pz_4=InterpolateMap(xm+1/3,ym+1/3);
        #local pz=(pz_1+pz_2+pz_3+pz_4)/4;
      #break
      
    #end
    
    <px,py,pz>
    
    #local jj=jj+1;
  #end
  #local ii=ii+1;
#end
  
    }
#end //HeightfieldMode=3

#if (HeightfieldMode=2)
        <X,Y,Z>,<X1,Y,Z+sx0>,<X2,Y,Zx-sx1>,<X3,Y,Zx>,
        <X,Y1,Z+sy0>,<X1,Y1,Z+sx0+sy0>,<X2,Y1,Zx-sx1+sy1>,<X3,Y1,Zx+sy1>,
        <X,Y2,Zy-sy2>,<X1,Y2,Zy+sx2-sy2>,<X2,Y2,Zxy-sx3-sy3>,<X3,Y2,Zxy-sy3>,
        <X,Y3,Zy>,<X1,Y3,Zy+sx2>,<X2,Y3,Zxy-sx3>,<X3,Y3,Zxy>
    }
#end // HeightfieldMode==2
  
#declare j=j+1; #declare Y=Y+dy;
#end
#declare i=i+1; #declare X=X+dx;
#end
  
#if (HeightfieldMode<2)
} //mesh
#end

#if (WithInfinitePlane)
difference
    {
    plane { z,0 }
    box { <0,0,1>,<field_World_wrldsiz,field_World_wrldsiz,-100> }
    }
#end    
    translate <0,0,OFFSET_Z>
    
  Ground0Texture()

  } //union
#end //#macro MakeSmoothWorld()
#end //#ifndef (MakeSmoothWorld)

#macro SubdivideMapData()
  #local NX=dimension_size(MapData,2);
  #local NY=dimension_size(MapData,1);

  #local TmpMapData=array[NX][NY];
  #local Y=0;
  #while(Y<NY)
    #local X=0;
    #while(X<NX)
      #local TmpMapData[X][Y]=MapData[X][Y];
      #local X=X+1;
    #end
    #local Y=Y+1;
  #end

  #declare MapData=array[2*NX-1][2*NY-1];
  #local Y=0;
  #while(Y<NY)
    #local X=0;
    #while(X<NX)
      #declare MapData[2*X][2*Y]=TmpMapData[X][Y];
      #if (X<(NX-1))
        #declare MapData[2*X+1][2*Y]=(TmpMapData[X][Y]+TmpMapData[X+1][Y])/2;
      #end
      #if (Y<(NY-1))
        #declare MapData[2*X][2*Y+1]=(TmpMapData[X][Y]+TmpMapData[X][Y+1])/2;
        #if (X<(NX-1))
//          #declare MapData[2*X+1][2*Y+1]=(TmpMapData[X][Y]+TmpMapData[X+1][Y]+TmpMapData[X][Y+1]+TmpMapData[X+1][Y+1])/4; // rectangular grid
          #declare MapData[2*X+1][2*Y+1]=(TmpMapData[X][Y]+TmpMapData[X+1][Y+1])/2; // triangular grid
        #end
      #end
      #local X=X+1;
    #end
    #local Y=Y+1;
  #end

#end
  
#macro EndWorld()
#switch (field_World_wrldbnd)
#case (BORDER_FENCE)
#if (field_World_wrldwat<=0)
Fence()
#else
UnderWaterStones()
#end
#break
#case (BORDER_TELEPORT)
TeleportBoundary()
#break
#end
#if (field_World_wrldtyp=WORLD_SMOOTH)
  #local N=HeightfieldSubdivision;
  #while(N>0)
    SubdivideMapData()
    #local N=N-1;
  #end
  #declare GroundObject=MakeSmoothWorld(true);
  object{GroundObject}
#end
#end

#include "math.inc"
// calculate camera location, to be used inside camera transform instead of the forbidden look_at
#macro LookAtRotation(dir)
  #local zdir=<dir.x,dir.y,0>;
  #local zangle=VAngleD(zdir,<0,1,0>);
  #if (dir.x<0)
    #local zangle=-zangle;
  #end
  #local dir2=vrotate(dir,<0,0,zangle>);
  #local xangle=VAngleD(dir2,<0,1,0>);
  #if (dir2.z>0)
    #local xangle=-xangle;
  #end
  <-xangle,0,-zangle>
#end
  
#macro CameraIsUnderwater()
 (CameraLocation.z < (field_World_wrldwat+OFFSET_Z))
#end
 
#declare NoCameraSet=1;
#declare CameraIgnoreRotation=false;
#macro Camera(Loc,LookAt)
#if (NoCameraSet)
#declare CameraLocation=Loc;
#declare CameraTarget=LookAt;
camera { StandardCamera
  #if (!CameraIgnoreRotation)
  rotate LookAtRotation(LookAt-Loc)
  #end
  translate Loc
  }
#declare NoCameraSet=0;
#end
#end

//////////////////////////////

#macro BeginScene()
  
#ifndef (SunPhase) //SunPhase=1 (default) = use LightColor and default sky colors, SunPhase=0..1 = use the built-in sunrise/set transition
 #declare SunPhase=1;
#end

#if (SunPhase<1)
 #if (SunPhase<0.5)
    #local w2=2*SunPhase;
    #local w1=1-w2;
#declare HorizonColor=HorizonColor_0*w1+HorizonColor_1*w2;
#declare MidSkyColor0=MidSkyColor0_0*w1+MidSkyColor0_1*w2;
#declare MidSkyColor1=MidSkyColor1_0*w1+MidSkyColor1_1*w2;
#declare MidSkyColor2=MidSkyColor2_0*w1+MidSkyColor2_1*w2;
#declare MidSkyColor3=MidSkyColor3_0*w1+MidSkyColor3_1*w2;
#declare HighSkyColor0=HighSkyColor0_0*w1+HighSkyColor0_1*w2;
#declare HighSkyColor1=HighSkyColor1_0*w1+HighSkyColor1_1*w2;
#declare HighSkyColor2=HighSkyColor2_0*w1+HighSkyColor2_1*w2;
#declare HighSkyColor3=HighSkyColor3_0*w1+HighSkyColor3_1*w2;
#declare LightColor=LightColor_0*w1+LightColor_1*w2;

 #else

  #local w2=2*(SunPhase-0.5);
  #local w1=1-w2;
#declare HorizonColor=HorizonColor_1*w1+HorizonColor_2*w2;
#declare MidSkyColor0=MidSkyColor0_1*w1+MidSkyColor0_2*w2;
#declare MidSkyColor1=MidSkyColor1_1*w1+MidSkyColor1_2*w2;
#declare MidSkyColor2=MidSkyColor2_1*w1+MidSkyColor2_2*w2;
#declare MidSkyColor3=MidSkyColor3_1*w1+MidSkyColor3_2*w2;
#declare HighSkyColor0=HighSkyColor0_1*w1+HighSkyColor0_2*w2;
#declare HighSkyColor1=HighSkyColor1_1*w1+HighSkyColor1_2*w2;
#declare HighSkyColor2=HighSkyColor2_1*w1+HighSkyColor2_2*w2;
#declare HighSkyColor3=HighSkyColor3_1*w1+HighSkyColor3_2*w2;
#declare LightColor=LightColor_1*w1+LightColor_2*w2;
 #end

#else // no SunPhase

#ifndef (HorizonColor)
 #declare HorizonColor=HorizonColor_2;
#end
#ifndef (MidSkyColor0)
 #declare MidSkyColor0=MidSkyColor0_2;
 #declare MidSkyColor1=MidSkyColor1_2;
 #declare MidSkyColor2=MidSkyColor2_2;
 #declare MidSkyColor3=MidSkyColor3_2;
 #declare HighSkyColor0=HighSkyColor0_2;
 #declare HighSkyColor1=HighSkyColor1_2;
 #declare HighSkyColor2=HighSkyColor2_2;
 #declare HighSkyColor3=HighSkyColor3_2;
 #declare LightColor=LightColor_2;
#end

#end

StandardSky()
StandardLight()

BeginEnergy()

#if (!CameraIsUnderwater())
  #declare FogAlt=max(field_World_wrldwat,0)+AirFogThickness+OFFSET_Z;
  #ifdef (AirFogDistance)
  fog { color srgb HorizonColor distance AirFogDistance fog_type 2  fog_offset FogAlt fog_alt 5 rotate <90,0,0> }
  #end
#end

#if ((field_World_wrldwat>0) | (field_World_wrldtyp != WORLD_FLAT))
// polygon {5, <-999,-999,0>,<999,-999,0>,<999,999,0>,<-999,999,0>,<-999,-999,0>
  WaterObject()
#end
  
#end //#macro BeginScene()

#macro EndScene()
EndEnergy()
#ifdef(EndSceneExtra)
 EndSceneExtra()
#end
#end

#macro BeginObjectStd()
#declare ObjectShift=<0,0,0>;
#declare CreatureUniqueSeed=seed(CreatureUIDAsNumber());
#declare PartAssimTab=array[field_Creature_numparts]
#declare PartRotationTab=array[field_Creature_numparts]
#if (!strcmp(field_m_Vstyle,FORAM_M_VSTYLE))
 BeginForam()
#end
#end

#ifndef (BeginObject)
#macro BeginObject()
  BeginObjectStd()
#end
#end

#macro EndObject()
#if (!strcmp(field_m_Vstyle,FORAM_M_VSTYLE))
 EndForam()
#end
#end

#declare CreatePartAtEnd=1;//valid between beginpart/endpart

//#################################################
  
#macro WaterObject()
#if (WaterTrueWaves>0)
  #local WaterMargin=WaterTrueWaves+0.1;
  isosurface
  {
  function { z+f_snoise3d(x/(20*WaterTrueWaves),y/(20*WaterTrueWaves),(z+0.2*AnimFrame)/(20*WaterTrueWaves))*WaterTrueWaves }
  threshold 0
  contained_by { box { <-WaterMargin,-WaterMargin,-50>,<field_World_wrldsiz+WaterMargin,field_World_wrldsiz+WaterMargin,WaterMargin> } }
  WaterTexture()
  WaterInterior()
  translate <0,0,field_World_wrldwat+OFFSET_Z> 
  }
#else
  plane { z,0
  WaterTexture()
  WaterInterior()
  translate <0,0,field_World_wrldwat+OFFSET_Z> 
  hollow
  }
#end

#if (CameraIsUnderwater())
 #ifdef(WaterFogColor)
 fog { color WaterFogColor
   distance WaterFogDistance
	fog_type 2
	fog_offset field_World_wrldwat+OFFSET_Z-3
	fog_alt 2
	rotate <90,0,0>
	}
  #end
#end
#end

//#################################################
  
#macro BeginPart(Id)
#declare St_Recept_S=0;
#declare St_Recept_T=0;
#declare CreatePartAtEnd=1;
#declare PartAssimTab[Id]=field_p_as;
#declare CurrentPartId=Id;
#end
  
#macro PartGeometryStd(Location,Rotation)
 #declare PartRotationTab[CurrentPartId]=Rotation;
 #if (!strcmp(field_m_Vstyle,"food"))
  Energy(Location,pow(field_Creature_energy/field_Creature_energy0,1/3),false)
  #declare CreatePartAtEnd=0;
 #elseif (!strcmp(field_m_Vstyle,"manipulator"))
  Manipulator(Location,0.0)
  #declare CreatePartAtEnd=0;
 #elseif (!strcmp(field_m_Vstyle,"nutrient"))
  NutrientShape(Location,Rotation)
  #declare CreatePartAtEnd=0;
 #elseif (!strcmp(field_m_Vstyle,"nutrient_visual"))
  #declare CreatePartAtEnd=0;
  //invisible
 #else
 #ifndef (field_p_sh)
  #declare field_p_sh=0;
 #end
 #switch (field_p_sh)
  #case (1)
   #if (!strcmp(field_m_Vstyle,FORAM_M_VSTYLE))
    ForamShape(Location,Rotation)
   #else
    EllipsoidShape(Location,Rotation)
   #end
   #declare CreatePartAtEnd=0;
  #break
  #case (2)
   CuboidShape(Location,Rotation)
   #declare CreatePartAtEnd=0;
  #break
  #case (3)
   #if (!strcmp(field_m_Vstyle,"reticulopodia"))
    ReticulopodiaShape(Location,Rotation)
   #else
    CylinderShape(Location,Rotation)
   #end
   #declare CreatePartAtEnd=0;
  #break
 #end
 #end
 #declare StickEndLoc=Location;
 #declare StickEndRot=Rotation;
#end

#ifndef (PartGeometry)
#macro PartGeometry(Location,Rotation)
  PartGeometryStd(Location,Rotation)
#end
#end
  
#macro EndPart()
#if (CreatePartAtEnd)
CreatureStickEnd(StickEndLoc,StickEndRot)
#end
#end

#macro BeginJointStd(Part1,Part2)
#declare St_Recept_G=0;
#declare St_BendMuscle=0;
#declare St_RotMuscle=0;
#declare St_Asimil=(PartAssimTab[Part1]+PartAssimTab[Part2])/2.0;
#declare CurrentJointP1=Part1;
#declare CurrentJointP2=Part2;
#end

#macro BeginJoint(Part1,Part2)
  BeginJointStd(Part1,Part2)
#end

#macro JointGeometry(Location1,Location2,Rotation)
  JointGeometryStd(Location1,Location2,Rotation)
#end

#macro JointGeometryStd(Location1,Location2,Rotation)
#declare StickLoc1=Location1;
#declare StickLoc2=Location2;
#declare StickRot=Rotation;
#end

#macro EndJointStd()
#declare St_Speed=0.5;
#ifndef (field_j_sh)
 #declare field_j_sh=0;
#end
#if (field_j_sh=0)
 CreatureStick(StickLoc1,StickLoc2,StickRot)
#end
#if (St_BendMuscle)
  object { RotMuscle() rotate <0,90,0> rotate PartRotationTab[CurrentJointP1]*57.3 translate StickLoc1 }
#end
#if (St_RotMuscle)
  object { RotMuscle() rotate PartRotationTab[CurrentJointP1]*57.3 translate StickLoc1 }
#end
#end

#macro EndJoint()
  EndJointStd()
#end

#macro BeginNeuro(NClass)
 #if (!strcmp(NClass,"|"))
  #declare St_BendMuscle=1;
  #declare St_BendMusclePos=field_n_s*3.14;
 #end
 #if (!strcmp(NClass,"@"))
  #declare St_RotMuscle=1;
 #end
 #if (!strcmp(NClass,"G"))
  #declare St_Recept_G=1;
 #end
 #if (!strcmp(NClass,"S"))
  #declare St_Recept_S=1;
 #end
 #if (!strcmp(NClass,"T"))
  #declare St_Recept_T=1;
 #end
#end

#macro EndNeuro()
#end

#macro BeginDust()
#end
  
#macro EndDust()
#end
  
#declare DustScatteringRGB=1.0;
#declare DustScatteringExtinction=1.0;
#declare DustAbsorptionRGB=1.0;
  
#macro Dust(Id,Age,Pos,Size,Density)
//sphere { Pos,Size hollow pigment { srgbt <0.2,0.2,0.2,1.0-Density*exp(-0.1*Age)> } }
#local S=Size*1.2;
#if ((Pos.z+S)>(field_World_wrldwat+OFFSET_Z)) //hide underwater dust
#local D=DustScatteringRGB*Density;
//  #if (D>0.0001)
    #local R=seed(Id); //individual turbulence pattern per particle
    #local X=rand(R)*20;
    #local Y=rand(R)*20;
    #local DustInterior= interior { media { 
//	                 method 2
//	                 samples 3
	                 scattering { 1, D extinction DustScatteringExtinction }
	                   absorption DustAbsorptionRGB*Density
	                 density { bozo scale 0.4 color_map { [0 rgb 0] [0.3 rgb 0] [0.7 rgb 1.0] [1 rgb 1] } translate <X,Y,0> }
	                 density { spherical color_map { [0 rgb 0] [0.4 rgb 1] [1 rgb 1] } }
	} };

    #local PartiallyUnderwater=((Pos.z-S)<(field_World_wrldwat+OFFSET_Z));
    #if (PartiallyUnderwater)
    difference {
    #end
    sphere
      { <0,0,0>,1.0 hollow pigment { color Clear }
	interior { DustInterior }
	scale S translate Pos
      }
    #if (PartiallyUnderwater)
    plane { z, 0 hollow pigment { color Clear }
	interior { DustInterior }
	translate field_World_wrldwat+OFFSET_Z+0.001
      }
    }
    #end
//  #end
#end
  
#end
