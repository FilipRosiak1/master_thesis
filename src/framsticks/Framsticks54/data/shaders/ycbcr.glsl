glcompat_attrs=4 //texture

[VERTEX]

#include "common.glsl"

IN vec4 a_vert;
IN vec4 a_texcoord;
OUT mediump vec2 v_texcoord;

void main()
{
    gl_Position = u_projmodmat * a_vert;
    v_texcoord=vec2(u_texmat*vec4(a_texcoord));
}

[FRAGMENT]

#include "common.glsl"

IN mediump vec2 v_texcoord;

void main()
{
    lowp vec4 ycbcr=vec4(texture2D(u_tex,v_texcoord).r,texture2D(u_tex2,v_texcoord).ra,1.0);
    const mediump mat4 ycbcr_to_rgb =
        mat4(vec4(+1.0000, +1.0000, +1.0000, +0.0000),
             vec4(+0.0000, -0.3441, +1.7720, +0.0000),
             vec4(+1.4020, -0.7141, +0.0000, +0.0000),
             vec4(-0.7010, +0.5291, -0.8860, +1.0000) );
    gl_FragColor = ycbcr_to_rgb * ycbcr;
}
