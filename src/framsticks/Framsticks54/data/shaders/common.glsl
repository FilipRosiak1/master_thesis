#ifndef OPENGL_ES
#define lowp
#define mediump
#define highp
#endif

#if __VERSION__ >= 300

 #ifdef VERTEX_SHADER
  #define IN in
  #define OUT out
 #endif
 #ifdef FRAGMENT_SHADER
  #define IN in
  out lowp vec4 outFragColor;
  #define gl_FragColor outFragColor
 #endif
 #define texture2D texture
 #define texture2DLod textureLod
 #define textureCube texture

#else

 #ifdef VERTEX_SHADER
  #define IN attribute
  #define OUT varying
 #endif
 #ifdef FRAGMENT_SHADER
  #define IN varying
 #endif

#endif
