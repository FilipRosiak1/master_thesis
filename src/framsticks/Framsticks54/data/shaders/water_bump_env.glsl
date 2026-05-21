glcompat_attrs=5
class_envprogram=1

[VERTEX]

#include "common.glsl"

IN vec4 a_vert;
IN vec3 a_norm;
IN vec4 a_texcoord;

OUT mediump vec3 v_pos;
OUT mediump vec3 v_wnorm;
OUT mediump vec3 v_lnorm;
OUT mediump vec2 v_texcoord;

void main()
{
	gl_Position = u_projmodmat * a_vert;
	vec3 ecnormal=u_rotmat*a_norm;
    	v_lnorm = normalize(u_rotmat * a_norm);
	v_wnorm = normalize(mat3(u_viewmat_inv) * v_lnorm);
	v_pos = (u_viewmat_inv * (u_modmat * a_vert)).xyz;
	v_texcoord=vec2(u_texmat*vec4(a_texcoord));
}

[FRAGMENT]

#include "common.glsl"

IN mediump vec3 v_pos;
IN mediump vec3 v_wnorm;
IN mediump vec2 v_texcoord;
IN mediump vec3 v_lnorm;

void main()
{
	mediump vec2 bump = texture2D(u_tex,v_texcoord).rg-vec2(0.5,0.5);
	mediump vec3 dir = u_campos - v_pos;
	mediump vec3 norm = normalize(v_wnorm+vec3(bump,0.0)*0.2);
	mediump vec3 refl = normalize(reflect(-dir, norm));
#ifdef REVERSE_SIDE
	refl.z = -refl.z;
#endif	
	mediump vec4 env = textureCube(u_envmap, refl);
	mediump float intensity = env.a*20.0;
	dir = normalize(dir);
	mediump float eye_norm_z = dot(dir,norm);
	eye_norm_z = pow(eye_norm_z, 0.5);
	lowp float refl_a = min(1.0, max(0.05, pow(1.0-eye_norm_z,0.5)*0.5)*intensity*1.5);
	//env.a = max(pow(1.0-eye_norm_z,0.5)*0.5, 0.05);
        lowp vec4 mixed = vec4(mix(u_color.rgb, env.rgb*intensity, refl_a/(refl_a+u_color.a)), min(1.0,refl_a+u_color.a));
	gl_FragColor = mixed;
}
