#pragma once

#define TO_STRING_(x) #x
#define TO_STRING(x) TO_STRING_(x)

#define DEMANDLOADINGGEOMETRY_CHUNK_CH_SHADER_NAME __closesthit__DemandLoadingGeomtry_chunk
#define DEMANDLOADINGGEOMETRY_CHUNK_CH_SHADER_NAME_STRING TO_STRING(DEMANDLOADINGGEOMETRY_CHUNK_CH_SHADER_NAME)