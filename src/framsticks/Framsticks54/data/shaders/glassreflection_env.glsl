glcompat_attrs=5
class_envprogram=1

[VERTEX]

#include "common.glsl"

IN vec4 a_vert;
IN vec3 a_norm;

OUT mediump vec3 v_pos;
OUT mediump vec3 v_lnorm;
OUT mediump vec3 v_wnorm;

void main()
{
	gl_Position = u_projmodmat * a_vert;
	vec3 ecnormal=u_rotmat*a_norm;
    	v_lnorm = normalize(u_rotmat * a_norm);
	v_wnorm = normalize(mat3(u_viewmat_inv) * v_lnorm);
	v_pos = (u_viewmat_inv * (u_modmat * a_vert)).xyz;
}

[FRAGMENT]

#include "common.glsl"

IN mediump vec3 v_pos;
IN mediump vec3 v_wnorm;
IN mediump vec2 v_texcoord;
IN mediump vec3 v_lnorm;

void main()
{
	mediump vec3 dir = u_campos - v_pos;
	mediump vec3 refl = normalize(reflect(-dir, v_wnorm));
	mediump vec4 env = textureCube(u_envmap, refl);
	mediump float intensity = env.a*20.0;
	lowp float refl_a = min(1.0, max(0.05, 1.0-v_lnorm.z)*intensity*1.5);
	lowp vec4 mixed = vec4(mix(u_color.rgb, env.rgb*intensity, refl_a/(refl_a+u_color.a)), min(1.0,refl_a+u_color.a));
	gl_FragColor = mixed;
}
