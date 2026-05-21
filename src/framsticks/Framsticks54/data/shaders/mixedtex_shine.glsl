glcompat_attrs=21 //FLAG_TEXTURE+FLAG_LIGHT+FLAG_SPECULAR
class_mixedtexture=1

[VERTEX]

#include "common.glsl"

IN mediump vec4 a_vert;
OUT mediump vec4 v_light;
IN vec3 a_norm;
OUT mediump vec3 v_hv;
OUT mediump vec3 v_ecnormal;
IN vec4 a_texcoord;
OUT mediump vec2 v_texcoord;

void main()
{
gl_Position = u_projmodmat * a_vert;
v_ecnormal=normalize(u_rotmat*a_norm);
v_light=max(dot(u_lightdir,v_ecnormal),0.0)*u_diffuse+u_ambient;
v_hv=normalize(-normalize(vec3(u_modmat*a_vert))+u_lightdir);
v_texcoord=vec2(u_texmat*vec4(a_texcoord));
}

[FRAGMENT]

#include "common.glsl"

IN mediump vec4 v_light;
IN mediump vec4 v_specular;
IN mediump vec2 v_texcoord;
IN mediump vec3 v_hv;
IN mediump vec3 v_ecnormal;

void main()
{
lowp vec4 c=u_color;
c*=mix(texture2D(u_tex,v_texcoord),texture2D(u_tex2,v_texcoord),u_mix);
c*=v_light;
mediump vec3 hv = normalize(v_hv);
mediump vec3 ecnormal = normalize(v_ecnormal);
lowp vec4 specular=u_speccolor*pow(max(dot(hv,ecnormal),0.0),u_shininess);
c+=specular;
gl_FragColor=c;
}
