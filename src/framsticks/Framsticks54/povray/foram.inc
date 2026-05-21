#declare DeadForamTextureScale=0.2;
#declare ForceDeadForamTexture=false;

#declare ForamRGBNormal=<1, 0.75, 0.4, 0.7>;
#declare ForamRGBTransparent=<0.85, 0.9, 0.95, 0.7>;
#declare ForamRGBMilky=<0.7, 0.7, 0.7, 0.15>;
#declare ForamRGBInterior=<1, 0.5, 0.2>;
#declare ForamInteriorR=0.92;

#macro GetForamGrowingState(State)
 #if(strlen(field_p_i)>8)
   #if(substr(field_p_i,1,8)="growing=")
     #declare State=max(0.0,min(1.0,val(substr(field_p_i,9,strlen(field_p_i)-8))));
   #end
 #end
#end

#macro ForamOuterShell(Location,Rotation,Scale,Growing)
  #if (Growing<=0.6)
   difference
     {
     sphere { <0,0,0>,1 hollow }
     cylinder { <-0.9,0,0>,<-1.1,0,0>,0.15 hollow }
     scale Scale
     rotate Rotation*57.3
     translate Location
     }
  #else
     #if (Growing<0.8)
       #local Inner=((Growing-0.6)/0.2);
     #else
       #local Inner=1;
     #end
     #local Inner=min(Inner,0.95);
     #local Inner=Inner*1.02*ForamInteriorR;
       difference { sphere { <0,0,0>,1 hollow }
	            sphere { <0,0,0>,Inner hollow }
                    cylinder { <-0.9,0,0>,<-1.1,0,0>,0.15 hollow }
	 scale Scale
	 rotate Rotation*57.3
	 translate Location
       }
  #end
#end

#macro ForamOuterShellCutout(Location,Rotation,Scale,Growing)
  #if (Growing>0.8)
     sphere { <0,0,0>,ForamInteriorR*1.01 hollow
     scale Scale
     rotate Rotation*57.3
     translate Location
     }
  #end
#end

#macro ForamInterior(Location,Rotation,Scale,Growing)
  #if(Growing<0.81)
    #local R=0.0001;
    sphere { <0,0,0>,ForamInteriorR*R hollow
    scale Scale
    rotate Rotation*57.3
    translate Location
   }
  #else
    #local R=(Growing-0.8)/0.2;
    difference {
      sphere { <0,0,0>,ForamInteriorR*R hollow }
      sphere { <-1,0,0>,ForamInteriorR*0.15 hollow }
    scale Scale
    rotate Rotation*57.3
    translate Location
    }
  #end
#end

#macro DeadForamTexture(bumpseed,tint)
#local bshift1=<rand(bumpseed)*10,rand(bumpseed)*10,rand(bumpseed)*10>;
#local bshift2=<rand(bumpseed)*10,rand(bumpseed)*10,rand(bumpseed)*10>;
texture {
  pigment { average pigment_map {
    
    [0.3 crackle form <1,0,0> color_map { [0 rgb <0,0,0>] [0.1 rgb <0,0,0>] [0.5 rgb <1,1,1>] } translate bshift1 ]

    [0.3 crackle form <1,0,0> color_map { [0 rgb <0,0,0>] [0.1 rgb <0,0,0>] [0.5 rgb <1,1,1>] } translate bshift2 ]

    [1.0 color tint ]
    
//    [1.0 bozo scale 6 translate tintshift color_map { [0 rgb <1,0.7,0.3>] [1 rgb <1,0.95,0.9>] }]
    
  } }
  normal { average normal_map {
    [1.0 crackle form <1,0,0> slope_map { [0 <-2,0>] [0.3 <-1,2>] [0.6 <0,0>] [1 <0,0>] } translate bshift1 ]
    [1.0 crackle form <1,0,0> slope_map { [0 <-2,0>] [0.3 <-1,2>] [0.6 <0,0>] [1 <0,0>] } translate bshift2 ]
      
      [0.3 bumps 0.8 scale 0.1] } }
  finish { phong 0.5 ambient 0.3/GlobalAmbient diffuse 0.9 }
  scale DeadForamTextureScale/6
}
#end

#macro DeadForamBlob(sc,bumpseed,tint)
  blob {
  threshold 1
  sphere { <0,0,0>,2,2 }
  cylinder { <0,0,0>,<2,0,0>,0.3,-0.3 }
  cylinder { <0,0,0>,<2,0,0>,0.15,-0.3 }
//  texture {pigment { bozo scale 0.6 color_map { [0 rgb <1,0.7,0.3>] [1 rgb <1,0.95,0.9>] }}}
  scale sc
  DeadForamTexture(bumpseed,tint)
}
#end

#macro ForamShapeSingle(Location,Rotation)
  object {
  #if((field_Creature_energy > 0.0) & (!ForceDeadForamTexture))
    #local Growing=1.0;
    GetForamGrowingState(Growing)
    ForamOuterShell(Location,Rotation,<field_p_sx,field_p_sy,field_p_sz>,Growing)
  #else
    object {
    DeadForamBlob(<field_p_sx,field_p_sy,field_p_sz>,ForamCreatureSeed,ForamTint)
    rotate Rotation*57.3 translate Location }
  #end
  }
#end

#macro ForamInteriorSingle(Location,Rotation)
  #local Growing=1.0;
  GetForamGrowingState(Growing)
  ForamInterior(Location,Rotation,<field_p_sx,field_p_sy,field_p_sz>,Growing)
#end

#macro ForamHoleSingle(Location,Rotation)
//  torus { 0.15,0.03 pigment { rgbf <0,0,0,0.01> } rotate <0,0,90> translate <0.94,0,0> interior { ior ForamIOR } finish { reflection { 0 } }
//  disc { <0.95,0,0>,<1,0,0>,0.2 pigment { rgbf <0,0,0,0.7> }
  cylinder { <0.95,0,0>,<0.9,0,0>,0.15 open hollow pigment { rgbf <0,0,0,0.7> } interior { ior ForamIOR }
  scale <field_p_sx,field_p_sy,field_p_sz>
  rotate Rotation*57.3
  translate Location }
#end

#declare MAX_FORAM_ELEMENTS=100;
#declare ForamElementLoc=array[MAX_FORAM_ELEMENTS]
#declare ForamElementRot=array[MAX_FORAM_ELEMENTS]
#declare ForamElementSca=array[MAX_FORAM_ELEMENTS]
#declare ForamElementGro=array[MAX_FORAM_ELEMENTS]
#declare ForamElementCount=0;
  
#macro ForamShape(Location,Rotation) 

  #if (field_Creature_center_z<50) //!!!
  
  #if((field_Creature_energy > 0.0) & (!ForceDeadForamTexture))
    #local Growing=1.0;
    GetForamGrowingState(Growing)
    #if (ForamElementCount<MAX_FORAM_ELEMENTS)
      #declare ForamElementLoc[ForamElementCount]=Location;
      #declare ForamElementRot[ForamElementCount]=Rotation;
      #declare ForamElementSca[ForamElementCount]=<field_p_sx,field_p_sy,field_p_sz>;
      #declare ForamElementGro[ForamElementCount]=Growing;
      #declare ForamElementCount=ForamElementCount+1;
    #end
    
  #else
    object {
    DeadForamBlob(<field_p_sx,field_p_sy,field_p_sz>,ForamCreatureSeed,ForamTint)
    rotate Rotation*57.3 translate Location }
  #end

  #end //!!!
    
#end //ForamShape

#macro BeginForam()
  #declare ForamElementCount=0;
  #declare ForamCreatureSeed=seed(CreatureUIDAsNumber());
  #local G=rand(ForamCreatureSeed)*0.6+0.1;
  #local S=rand(ForamCreatureSeed)*0.6+0.1;
  #declare ForamTint=<1,G+(1.0-G)*S,S>;
//  #debug concat("tint=",vstr(3, ForamTint," / ",0,3),"\n")
//  #declare ForamTint=<1,0.8,0.5>;
#end
  
#macro EndForam()

  #if (ForamElementCount>0)
  difference
    {
  merge
    {
    #local E=0;
    #while (E<ForamElementCount)
      ForamOuterShell(ForamElementLoc[E],ForamElementRot[E],ForamElementSca[E],ForamElementGro[E])
      #local E=E+1;
    #end
    } //merge
    #local E=0;
    #while (E<ForamElementCount)
      ForamOuterShellCutout(ForamElementLoc[E],ForamElementRot[E],ForamElementSca[E],ForamElementGro[E])
      #local E=E+1;
    #end
    finish { ambient 0.0 phong 0.3 phong_size 2 }
    pigment { color rgbt <1,1,1,0.9> }
    interior
      { ior ForamIOR
        media
        {
	  scattering { 1,1*8 extinction 0.4 }
//	  emission <1,0.7,0.6>*0.15
	  emission <1,0.2,0.0>*1
        }
      }      
    } //difference
  #local E=0;
  merge
    {
    #while (E<ForamElementCount)
      ForamInterior(ForamElementLoc[E],ForamElementRot[E],ForamElementSca[E],ForamElementGro[E])
      #local E=E+1;
    #end
    finish {  ambient <0.5,0.2,0.1>*2
           }
    interior { ior ForamIOR
                    media
	            { 
		      scattering { 1,1*3 extinction 1 }
		      emission ForamRGBInterior*2
		      absorption (<1,1,1>-ForamRGBInterior)*15
                    }
              }
    pigment { color rgbt 1 }
    } //merge
  #end //ForamElementCount>0
  
#end //EndForam()

