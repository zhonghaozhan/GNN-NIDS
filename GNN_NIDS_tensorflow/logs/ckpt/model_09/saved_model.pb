╒Ї9
ч╖
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
А
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resourceИ
A
EnsureShape

input"T
output"T"
shapeshape"	
Ttype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
о
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
)
Rank

input"T

output"	
Ttype
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_typeКэout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
М
SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
─
UnsortedSegmentSum	
data"T
segment_ids"Tindices
num_segments"Tnumsegments
output"T""
Ttype:
2	"
Tindicestype:
2	" 
Tnumsegmentstype0:
2	
░
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.15.02v2.15.0-rc1-8-g6887368d6d48╥ф5
з
weights_intermediateVarHandleOp*
_output_shapes
: *%

debug_nameweights_intermediate/*
dtype0*
shape:*%
shared_nameweights_intermediate
y
(weights_intermediate/Read/ReadVariableOpReadVariableOpweights_intermediate*
_output_shapes
:*
dtype0
Ш
false_negativesVarHandleOp*
_output_shapes
: * 

debug_namefalse_negatives/*
dtype0*
shape:* 
shared_namefalse_negatives
o
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
:*
dtype0
Ш
false_positivesVarHandleOp*
_output_shapes
: * 

debug_namefalse_positives/*
dtype0*
shape:* 
shared_namefalse_positives
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
:*
dtype0
Х
true_positivesVarHandleOp*
_output_shapes
: *

debug_nametrue_positives/*
dtype0*
shape:*
shared_nametrue_positives
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0
н
weights_intermediate_1VarHandleOp*
_output_shapes
: *'

debug_nameweights_intermediate_1/*
dtype0*
shape:*'
shared_nameweights_intermediate_1
}
*weights_intermediate_1/Read/ReadVariableOpReadVariableOpweights_intermediate_1*
_output_shapes
:*
dtype0
Ю
false_negatives_1VarHandleOp*
_output_shapes
: *"

debug_namefalse_negatives_1/*
dtype0*
shape:*"
shared_namefalse_negatives_1
s
%false_negatives_1/Read/ReadVariableOpReadVariableOpfalse_negatives_1*
_output_shapes
:*
dtype0
Ю
false_positives_1VarHandleOp*
_output_shapes
: *"

debug_namefalse_positives_1/*
dtype0*
shape:*"
shared_namefalse_positives_1
s
%false_positives_1/Read/ReadVariableOpReadVariableOpfalse_positives_1*
_output_shapes
:*
dtype0
Ы
true_positives_1VarHandleOp*
_output_shapes
: *!

debug_nametrue_positives_1/*
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
Ю
false_positives_2VarHandleOp*
_output_shapes
: *"

debug_namefalse_positives_2/*
dtype0*
shape:*"
shared_namefalse_positives_2
s
%false_positives_2/Read/ReadVariableOpReadVariableOpfalse_positives_2*
_output_shapes
:*
dtype0
Ы
true_positives_2VarHandleOp*
_output_shapes
: *!

debug_nametrue_positives_2/*
dtype0*
shape:*!
shared_nametrue_positives_2
q
$true_positives_2/Read/ReadVariableOpReadVariableOptrue_positives_2*
_output_shapes
:*
dtype0
Ю
false_negatives_2VarHandleOp*
_output_shapes
: *"

debug_namefalse_negatives_2/*
dtype0*
shape:*"
shared_namefalse_negatives_2
s
%false_negatives_2/Read/ReadVariableOpReadVariableOpfalse_negatives_2*
_output_shapes
:*
dtype0
Ы
true_positives_3VarHandleOp*
_output_shapes
: *!

debug_nametrue_positives_3/*
dtype0*
shape:*!
shared_nametrue_positives_3
q
$true_positives_3/Read/ReadVariableOpReadVariableOptrue_positives_3*
_output_shapes
:*
dtype0
Ю
false_positives_3VarHandleOp*
_output_shapes
: *"

debug_namefalse_positives_3/*
dtype0*
shape:*"
shared_namefalse_positives_3
s
%false_positives_3/Read/ReadVariableOpReadVariableOpfalse_positives_3*
_output_shapes
:*
dtype0
Ы
true_positives_4VarHandleOp*
_output_shapes
: *!

debug_nametrue_positives_4/*
dtype0*
shape:*!
shared_nametrue_positives_4
q
$true_positives_4/Read/ReadVariableOpReadVariableOptrue_positives_4*
_output_shapes
:*
dtype0
Ю
false_negatives_3VarHandleOp*
_output_shapes
: *"

debug_namefalse_negatives_3/*
dtype0*
shape:*"
shared_namefalse_negatives_3
s
%false_negatives_3/Read/ReadVariableOpReadVariableOpfalse_negatives_3*
_output_shapes
:*
dtype0
Ы
true_positives_5VarHandleOp*
_output_shapes
: *!

debug_nametrue_positives_5/*
dtype0*
shape:*!
shared_nametrue_positives_5
q
$true_positives_5/Read/ReadVariableOpReadVariableOptrue_positives_5*
_output_shapes
:*
dtype0
Ю
false_positives_4VarHandleOp*
_output_shapes
: *"

debug_namefalse_positives_4/*
dtype0*
shape:*"
shared_namefalse_positives_4
s
%false_positives_4/Read/ReadVariableOpReadVariableOpfalse_positives_4*
_output_shapes
:*
dtype0
Ы
true_positives_6VarHandleOp*
_output_shapes
: *!

debug_nametrue_positives_6/*
dtype0*
shape:*!
shared_nametrue_positives_6
q
$true_positives_6/Read/ReadVariableOpReadVariableOptrue_positives_6*
_output_shapes
:*
dtype0
Ю
false_negatives_4VarHandleOp*
_output_shapes
: *"

debug_namefalse_negatives_4/*
dtype0*
shape:*"
shared_namefalse_negatives_4
s
%false_negatives_4/Read/ReadVariableOpReadVariableOpfalse_negatives_4*
_output_shapes
:*
dtype0
Ы
true_positives_7VarHandleOp*
_output_shapes
: *!

debug_nametrue_positives_7/*
dtype0*
shape:*!
shared_nametrue_positives_7
q
$true_positives_7/Read/ReadVariableOpReadVariableOptrue_positives_7*
_output_shapes
:*
dtype0
Ю
false_positives_5VarHandleOp*
_output_shapes
: *"

debug_namefalse_positives_5/*
dtype0*
shape:*"
shared_namefalse_positives_5
s
%false_positives_5/Read/ReadVariableOpReadVariableOpfalse_positives_5*
_output_shapes
:*
dtype0
Ы
true_positives_8VarHandleOp*
_output_shapes
: *!

debug_nametrue_positives_8/*
dtype0*
shape:*!
shared_nametrue_positives_8
q
$true_positives_8/Read/ReadVariableOpReadVariableOptrue_positives_8*
_output_shapes
:*
dtype0
Ю
false_negatives_5VarHandleOp*
_output_shapes
: *"

debug_namefalse_negatives_5/*
dtype0*
shape:*"
shared_namefalse_negatives_5
s
%false_negatives_5/Read/ReadVariableOpReadVariableOpfalse_negatives_5*
_output_shapes
:*
dtype0
Ы
true_positives_9VarHandleOp*
_output_shapes
: *!

debug_nametrue_positives_9/*
dtype0*
shape:*!
shared_nametrue_positives_9
q
$true_positives_9/Read/ReadVariableOpReadVariableOptrue_positives_9*
_output_shapes
:*
dtype0
Ю
false_positives_6VarHandleOp*
_output_shapes
: *"

debug_namefalse_positives_6/*
dtype0*
shape:*"
shared_namefalse_positives_6
s
%false_positives_6/Read/ReadVariableOpReadVariableOpfalse_positives_6*
_output_shapes
:*
dtype0
Ю
true_positives_10VarHandleOp*
_output_shapes
: *"

debug_nametrue_positives_10/*
dtype0*
shape:*"
shared_nametrue_positives_10
s
%true_positives_10/Read/ReadVariableOpReadVariableOptrue_positives_10*
_output_shapes
:*
dtype0
Ю
false_negatives_6VarHandleOp*
_output_shapes
: *"

debug_namefalse_negatives_6/*
dtype0*
shape:*"
shared_namefalse_negatives_6
s
%false_negatives_6/Read/ReadVariableOpReadVariableOpfalse_negatives_6*
_output_shapes
:*
dtype0
Ю
true_positives_11VarHandleOp*
_output_shapes
: *"

debug_nametrue_positives_11/*
dtype0*
shape:*"
shared_nametrue_positives_11
s
%true_positives_11/Read/ReadVariableOpReadVariableOptrue_positives_11*
_output_shapes
:*
dtype0
Ю
false_positives_7VarHandleOp*
_output_shapes
: *"

debug_namefalse_positives_7/*
dtype0*
shape:*"
shared_namefalse_positives_7
s
%false_positives_7/Read/ReadVariableOpReadVariableOpfalse_positives_7*
_output_shapes
:*
dtype0
Ю
true_positives_12VarHandleOp*
_output_shapes
: *"

debug_nametrue_positives_12/*
dtype0*
shape:*"
shared_nametrue_positives_12
s
%true_positives_12/Read/ReadVariableOpReadVariableOptrue_positives_12*
_output_shapes
:*
dtype0
Ю
false_negatives_7VarHandleOp*
_output_shapes
: *"

debug_namefalse_negatives_7/*
dtype0*
shape:*"
shared_namefalse_negatives_7
s
%false_negatives_7/Read/ReadVariableOpReadVariableOpfalse_negatives_7*
_output_shapes
:*
dtype0
Ю
true_positives_13VarHandleOp*
_output_shapes
: *"

debug_nametrue_positives_13/*
dtype0*
shape:*"
shared_nametrue_positives_13
s
%true_positives_13/Read/ReadVariableOpReadVariableOptrue_positives_13*
_output_shapes
:*
dtype0
Ю
false_positives_8VarHandleOp*
_output_shapes
: *"

debug_namefalse_positives_8/*
dtype0*
shape:*"
shared_namefalse_positives_8
s
%false_positives_8/Read/ReadVariableOpReadVariableOpfalse_positives_8*
_output_shapes
:*
dtype0
Ю
true_positives_14VarHandleOp*
_output_shapes
: *"

debug_nametrue_positives_14/*
dtype0*
shape:*"
shared_nametrue_positives_14
s
%true_positives_14/Read/ReadVariableOpReadVariableOptrue_positives_14*
_output_shapes
:*
dtype0
Ю
false_negatives_8VarHandleOp*
_output_shapes
: *"

debug_namefalse_negatives_8/*
dtype0*
shape:*"
shared_namefalse_negatives_8
s
%false_negatives_8/Read/ReadVariableOpReadVariableOpfalse_negatives_8*
_output_shapes
:*
dtype0
Ю
true_positives_15VarHandleOp*
_output_shapes
: *"

debug_nametrue_positives_15/*
dtype0*
shape:*"
shared_nametrue_positives_15
s
%true_positives_15/Read/ReadVariableOpReadVariableOptrue_positives_15*
_output_shapes
:*
dtype0
Ю
false_positives_9VarHandleOp*
_output_shapes
: *"

debug_namefalse_positives_9/*
dtype0*
shape:*"
shared_namefalse_positives_9
s
%false_positives_9/Read/ReadVariableOpReadVariableOpfalse_positives_9*
_output_shapes
:*
dtype0
Ю
true_positives_16VarHandleOp*
_output_shapes
: *"

debug_nametrue_positives_16/*
dtype0*
shape:*"
shared_nametrue_positives_16
s
%true_positives_16/Read/ReadVariableOpReadVariableOptrue_positives_16*
_output_shapes
:*
dtype0
Ю
false_negatives_9VarHandleOp*
_output_shapes
: *"

debug_namefalse_negatives_9/*
dtype0*
shape:*"
shared_namefalse_negatives_9
s
%false_negatives_9/Read/ReadVariableOpReadVariableOpfalse_negatives_9*
_output_shapes
:*
dtype0
Ю
true_positives_17VarHandleOp*
_output_shapes
: *"

debug_nametrue_positives_17/*
dtype0*
shape:*"
shared_nametrue_positives_17
s
%true_positives_17/Read/ReadVariableOpReadVariableOptrue_positives_17*
_output_shapes
:*
dtype0
б
false_positives_10VarHandleOp*
_output_shapes
: *#

debug_namefalse_positives_10/*
dtype0*
shape:*#
shared_namefalse_positives_10
u
&false_positives_10/Read/ReadVariableOpReadVariableOpfalse_positives_10*
_output_shapes
:*
dtype0
Ю
true_positives_18VarHandleOp*
_output_shapes
: *"

debug_nametrue_positives_18/*
dtype0*
shape:*"
shared_nametrue_positives_18
s
%true_positives_18/Read/ReadVariableOpReadVariableOptrue_positives_18*
_output_shapes
:*
dtype0
б
false_negatives_10VarHandleOp*
_output_shapes
: *#

debug_namefalse_negatives_10/*
dtype0*
shape:*#
shared_namefalse_negatives_10
u
&false_negatives_10/Read/ReadVariableOpReadVariableOpfalse_negatives_10*
_output_shapes
:*
dtype0
Ю
true_positives_19VarHandleOp*
_output_shapes
: *"

debug_nametrue_positives_19/*
dtype0*
shape:*"
shared_nametrue_positives_19
s
%true_positives_19/Read/ReadVariableOpReadVariableOptrue_positives_19*
_output_shapes
:*
dtype0
б
false_positives_11VarHandleOp*
_output_shapes
: *#

debug_namefalse_positives_11/*
dtype0*
shape:*#
shared_namefalse_positives_11
u
&false_positives_11/Read/ReadVariableOpReadVariableOpfalse_positives_11*
_output_shapes
:*
dtype0
Ю
true_positives_20VarHandleOp*
_output_shapes
: *"

debug_nametrue_positives_20/*
dtype0*
shape:*"
shared_nametrue_positives_20
s
%true_positives_20/Read/ReadVariableOpReadVariableOptrue_positives_20*
_output_shapes
:*
dtype0
б
false_negatives_11VarHandleOp*
_output_shapes
: *#

debug_namefalse_negatives_11/*
dtype0*
shape:*#
shared_namefalse_negatives_11
u
&false_negatives_11/Read/ReadVariableOpReadVariableOpfalse_negatives_11*
_output_shapes
:*
dtype0
Ю
true_positives_21VarHandleOp*
_output_shapes
: *"

debug_nametrue_positives_21/*
dtype0*
shape:*"
shared_nametrue_positives_21
s
%true_positives_21/Read/ReadVariableOpReadVariableOptrue_positives_21*
_output_shapes
:*
dtype0
б
false_positives_12VarHandleOp*
_output_shapes
: *#

debug_namefalse_positives_12/*
dtype0*
shape:*#
shared_namefalse_positives_12
u
&false_positives_12/Read/ReadVariableOpReadVariableOpfalse_positives_12*
_output_shapes
:*
dtype0
Ю
true_positives_22VarHandleOp*
_output_shapes
: *"

debug_nametrue_positives_22/*
dtype0*
shape:*"
shared_nametrue_positives_22
s
%true_positives_22/Read/ReadVariableOpReadVariableOptrue_positives_22*
_output_shapes
:*
dtype0
б
false_negatives_12VarHandleOp*
_output_shapes
: *#

debug_namefalse_negatives_12/*
dtype0*
shape:*#
shared_namefalse_negatives_12
u
&false_negatives_12/Read/ReadVariableOpReadVariableOpfalse_negatives_12*
_output_shapes
:*
dtype0
Ю
true_positives_23VarHandleOp*
_output_shapes
: *"

debug_nametrue_positives_23/*
dtype0*
shape:*"
shared_nametrue_positives_23
s
%true_positives_23/Read/ReadVariableOpReadVariableOptrue_positives_23*
_output_shapes
:*
dtype0
б
false_positives_13VarHandleOp*
_output_shapes
: *#

debug_namefalse_positives_13/*
dtype0*
shape:*#
shared_namefalse_positives_13
u
&false_positives_13/Read/ReadVariableOpReadVariableOpfalse_positives_13*
_output_shapes
:*
dtype0
Ю
true_positives_24VarHandleOp*
_output_shapes
: *"

debug_nametrue_positives_24/*
dtype0*
shape:*"
shared_nametrue_positives_24
s
%true_positives_24/Read/ReadVariableOpReadVariableOptrue_positives_24*
_output_shapes
:*
dtype0
б
false_negatives_13VarHandleOp*
_output_shapes
: *#

debug_namefalse_negatives_13/*
dtype0*
shape:*#
shared_namefalse_negatives_13
u
&false_negatives_13/Read/ReadVariableOpReadVariableOpfalse_negatives_13*
_output_shapes
:*
dtype0
Ю
true_positives_25VarHandleOp*
_output_shapes
: *"

debug_nametrue_positives_25/*
dtype0*
shape:*"
shared_nametrue_positives_25
s
%true_positives_25/Read/ReadVariableOpReadVariableOptrue_positives_25*
_output_shapes
:*
dtype0
б
false_positives_14VarHandleOp*
_output_shapes
: *#

debug_namefalse_positives_14/*
dtype0*
shape:*#
shared_namefalse_positives_14
u
&false_positives_14/Read/ReadVariableOpReadVariableOpfalse_positives_14*
_output_shapes
:*
dtype0
Ю
true_positives_26VarHandleOp*
_output_shapes
: *"

debug_nametrue_positives_26/*
dtype0*
shape:*"
shared_nametrue_positives_26
s
%true_positives_26/Read/ReadVariableOpReadVariableOptrue_positives_26*
_output_shapes
:*
dtype0
б
false_negatives_14VarHandleOp*
_output_shapes
: *#

debug_namefalse_negatives_14/*
dtype0*
shape:*#
shared_namefalse_negatives_14
u
&false_negatives_14/Read/ReadVariableOpReadVariableOpfalse_negatives_14*
_output_shapes
:*
dtype0
Ю
true_positives_27VarHandleOp*
_output_shapes
: *"

debug_nametrue_positives_27/*
dtype0*
shape:*"
shared_nametrue_positives_27
s
%true_positives_27/Read/ReadVariableOpReadVariableOptrue_positives_27*
_output_shapes
:*
dtype0
б
false_positives_15VarHandleOp*
_output_shapes
: *#

debug_namefalse_positives_15/*
dtype0*
shape:*#
shared_namefalse_positives_15
u
&false_positives_15/Read/ReadVariableOpReadVariableOpfalse_positives_15*
_output_shapes
:*
dtype0
Ю
true_positives_28VarHandleOp*
_output_shapes
: *"

debug_nametrue_positives_28/*
dtype0*
shape:*"
shared_nametrue_positives_28
s
%true_positives_28/Read/ReadVariableOpReadVariableOptrue_positives_28*
_output_shapes
:*
dtype0
б
false_negatives_15VarHandleOp*
_output_shapes
: *#

debug_namefalse_negatives_15/*
dtype0*
shape:*#
shared_namefalse_negatives_15
u
&false_negatives_15/Read/ReadVariableOpReadVariableOpfalse_negatives_15*
_output_shapes
:*
dtype0
Ю
true_positives_29VarHandleOp*
_output_shapes
: *"

debug_nametrue_positives_29/*
dtype0*
shape:*"
shared_nametrue_positives_29
s
%true_positives_29/Read/ReadVariableOpReadVariableOptrue_positives_29*
_output_shapes
:*
dtype0
б
false_positives_16VarHandleOp*
_output_shapes
: *#

debug_namefalse_positives_16/*
dtype0*
shape:*#
shared_namefalse_positives_16
u
&false_positives_16/Read/ReadVariableOpReadVariableOpfalse_positives_16*
_output_shapes
:*
dtype0
Ю
true_positives_30VarHandleOp*
_output_shapes
: *"

debug_nametrue_positives_30/*
dtype0*
shape:*"
shared_nametrue_positives_30
s
%true_positives_30/Read/ReadVariableOpReadVariableOptrue_positives_30*
_output_shapes
:*
dtype0
б
false_negatives_16VarHandleOp*
_output_shapes
: *#

debug_namefalse_negatives_16/*
dtype0*
shape:*#
shared_namefalse_negatives_16
u
&false_negatives_16/Read/ReadVariableOpReadVariableOpfalse_negatives_16*
_output_shapes
:*
dtype0
Ю
true_positives_31VarHandleOp*
_output_shapes
: *"

debug_nametrue_positives_31/*
dtype0*
shape:*"
shared_nametrue_positives_31
s
%true_positives_31/Read/ReadVariableOpReadVariableOptrue_positives_31*
_output_shapes
:*
dtype0
в
false_negatives_17VarHandleOp*
_output_shapes
: *#

debug_namefalse_negatives_17/*
dtype0*
shape:╚*#
shared_namefalse_negatives_17
v
&false_negatives_17/Read/ReadVariableOpReadVariableOpfalse_negatives_17*
_output_shapes	
:╚*
dtype0
в
false_positives_17VarHandleOp*
_output_shapes
: *#

debug_namefalse_positives_17/*
dtype0*
shape:╚*#
shared_namefalse_positives_17
v
&false_positives_17/Read/ReadVariableOpReadVariableOpfalse_positives_17*
_output_shapes	
:╚*
dtype0
Ц
true_negativesVarHandleOp*
_output_shapes
: *

debug_nametrue_negatives/*
dtype0*
shape:╚*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:╚*
dtype0
Я
true_positives_32VarHandleOp*
_output_shapes
: *"

debug_nametrue_positives_32/*
dtype0*
shape:╚*"
shared_nametrue_positives_32
t
%true_positives_32/Read/ReadVariableOpReadVariableOptrue_positives_32*
_output_shapes	
:╚*
dtype0
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
|
count_1VarHandleOp*
_output_shapes
: *

debug_name
count_1/*
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
|
total_1VarHandleOp*
_output_shapes
: *

debug_name
total_1/*
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
д
Adam/v/dense_4/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/dense_4/bias/*
dtype0*
shape:*$
shared_nameAdam/v/dense_4/bias
w
'Adam/v/dense_4/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_4/bias*
_output_shapes
:*
dtype0
д
Adam/m/dense_4/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/dense_4/bias/*
dtype0*
shape:*$
shared_nameAdam/m/dense_4/bias
w
'Adam/m/dense_4/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_4/bias*
_output_shapes
:*
dtype0
о
Adam/v/dense_4/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/dense_4/kernel/*
dtype0*
shape
:@*&
shared_nameAdam/v/dense_4/kernel

)Adam/v/dense_4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_4/kernel*
_output_shapes

:@*
dtype0
о
Adam/m/dense_4/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/dense_4/kernel/*
dtype0*
shape
:@*&
shared_nameAdam/m/dense_4/kernel

)Adam/m/dense_4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_4/kernel*
_output_shapes

:@*
dtype0
д
Adam/v/dense_3/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/dense_3/bias/*
dtype0*
shape:@*$
shared_nameAdam/v/dense_3/bias
w
'Adam/v/dense_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_3/bias*
_output_shapes
:@*
dtype0
д
Adam/m/dense_3/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/dense_3/bias/*
dtype0*
shape:@*$
shared_nameAdam/m/dense_3/bias
w
'Adam/m/dense_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_3/bias*
_output_shapes
:@*
dtype0
п
Adam/v/dense_3/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/dense_3/kernel/*
dtype0*
shape:	А@*&
shared_nameAdam/v/dense_3/kernel
А
)Adam/v/dense_3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_3/kernel*
_output_shapes
:	А@*
dtype0
п
Adam/m/dense_3/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/dense_3/kernel/*
dtype0*
shape:	А@*&
shared_nameAdam/m/dense_3/kernel
А
)Adam/m/dense_3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_3/kernel*
_output_shapes
:	А@*
dtype0
е
Adam/v/dense_2/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/dense_2/bias/*
dtype0*
shape:А*$
shared_nameAdam/v/dense_2/bias
x
'Adam/v/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/bias*
_output_shapes	
:А*
dtype0
е
Adam/m/dense_2/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/dense_2/bias/*
dtype0*
shape:А*$
shared_nameAdam/m/dense_2/bias
x
'Adam/m/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/bias*
_output_shapes	
:А*
dtype0
░
Adam/v/dense_2/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/dense_2/kernel/*
dtype0*
shape:
АА*&
shared_nameAdam/v/dense_2/kernel
Б
)Adam/v/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_2/kernel* 
_output_shapes
:
АА*
dtype0
░
Adam/m/dense_2/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/dense_2/kernel/*
dtype0*
shape:
АА*&
shared_nameAdam/m/dense_2/kernel
Б
)Adam/m/dense_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_2/kernel* 
_output_shapes
:
АА*
dtype0
е
Adam/v/dense_1/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/dense_1/bias/*
dtype0*
shape:А*$
shared_nameAdam/v/dense_1/bias
x
'Adam/v/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/bias*
_output_shapes	
:А*
dtype0
е
Adam/m/dense_1/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/dense_1/bias/*
dtype0*
shape:А*$
shared_nameAdam/m/dense_1/bias
x
'Adam/m/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/bias*
_output_shapes	
:А*
dtype0
░
Adam/v/dense_1/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/dense_1/kernel/*
dtype0*
shape:
АА*&
shared_nameAdam/v/dense_1/kernel
Б
)Adam/v/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/kernel* 
_output_shapes
:
АА*
dtype0
░
Adam/m/dense_1/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/dense_1/kernel/*
dtype0*
shape:
АА*&
shared_nameAdam/m/dense_1/kernel
Б
)Adam/m/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/kernel* 
_output_shapes
:
АА*
dtype0
Я
Adam/v/dense/biasVarHandleOp*
_output_shapes
: *"

debug_nameAdam/v/dense/bias/*
dtype0*
shape:А*"
shared_nameAdam/v/dense/bias
t
%Adam/v/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense/bias*
_output_shapes	
:А*
dtype0
Я
Adam/m/dense/biasVarHandleOp*
_output_shapes
: *"

debug_nameAdam/m/dense/bias/*
dtype0*
shape:А*"
shared_nameAdam/m/dense/bias
t
%Adam/m/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense/bias*
_output_shapes	
:А*
dtype0
к
Adam/v/dense/kernelVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/dense/kernel/*
dtype0*
shape:
АА*$
shared_nameAdam/v/dense/kernel
}
'Adam/v/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense/kernel* 
_output_shapes
:
АА*
dtype0
к
Adam/m/dense/kernelVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/dense/kernel/*
dtype0*
shape:
АА*$
shared_nameAdam/m/dense/kernel
}
'Adam/m/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense/kernel* 
_output_shapes
:
АА*
dtype0
╟
Adam/v/update_connection/biasVarHandleOp*
_output_shapes
: *.

debug_name Adam/v/update_connection/bias/*
dtype0*
shape:	А*.
shared_nameAdam/v/update_connection/bias
Р
1Adam/v/update_connection/bias/Read/ReadVariableOpReadVariableOpAdam/v/update_connection/bias*
_output_shapes
:	А*
dtype0
╟
Adam/m/update_connection/biasVarHandleOp*
_output_shapes
: *.

debug_name Adam/m/update_connection/bias/*
dtype0*
shape:	А*.
shared_nameAdam/m/update_connection/bias
Р
1Adam/m/update_connection/bias/Read/ReadVariableOpReadVariableOpAdam/m/update_connection/bias*
_output_shapes
:	А*
dtype0
ь
)Adam/v/update_connection/recurrent_kernelVarHandleOp*
_output_shapes
: *:

debug_name,*Adam/v/update_connection/recurrent_kernel/*
dtype0*
shape:
АА*:
shared_name+)Adam/v/update_connection/recurrent_kernel
й
=Adam/v/update_connection/recurrent_kernel/Read/ReadVariableOpReadVariableOp)Adam/v/update_connection/recurrent_kernel* 
_output_shapes
:
АА*
dtype0
ь
)Adam/m/update_connection/recurrent_kernelVarHandleOp*
_output_shapes
: *:

debug_name,*Adam/m/update_connection/recurrent_kernel/*
dtype0*
shape:
АА*:
shared_name+)Adam/m/update_connection/recurrent_kernel
й
=Adam/m/update_connection/recurrent_kernel/Read/ReadVariableOpReadVariableOp)Adam/m/update_connection/recurrent_kernel* 
_output_shapes
:
АА*
dtype0
╬
Adam/v/update_connection/kernelVarHandleOp*
_output_shapes
: *0

debug_name" Adam/v/update_connection/kernel/*
dtype0*
shape:
АА*0
shared_name!Adam/v/update_connection/kernel
Х
3Adam/v/update_connection/kernel/Read/ReadVariableOpReadVariableOpAdam/v/update_connection/kernel* 
_output_shapes
:
АА*
dtype0
╬
Adam/m/update_connection/kernelVarHandleOp*
_output_shapes
: *0

debug_name" Adam/m/update_connection/kernel/*
dtype0*
shape:
АА*0
shared_name!Adam/m/update_connection/kernel
Х
3Adam/m/update_connection/kernel/Read/ReadVariableOpReadVariableOpAdam/m/update_connection/kernel* 
_output_shapes
:
АА*
dtype0
п
Adam/v/update_ip/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/update_ip/bias/*
dtype0*
shape:	А*&
shared_nameAdam/v/update_ip/bias
А
)Adam/v/update_ip/bias/Read/ReadVariableOpReadVariableOpAdam/v/update_ip/bias*
_output_shapes
:	А*
dtype0
п
Adam/m/update_ip/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/update_ip/bias/*
dtype0*
shape:	А*&
shared_nameAdam/m/update_ip/bias
А
)Adam/m/update_ip/bias/Read/ReadVariableOpReadVariableOpAdam/m/update_ip/bias*
_output_shapes
:	А*
dtype0
╘
!Adam/v/update_ip/recurrent_kernelVarHandleOp*
_output_shapes
: *2

debug_name$"Adam/v/update_ip/recurrent_kernel/*
dtype0*
shape:
АА*2
shared_name#!Adam/v/update_ip/recurrent_kernel
Щ
5Adam/v/update_ip/recurrent_kernel/Read/ReadVariableOpReadVariableOp!Adam/v/update_ip/recurrent_kernel* 
_output_shapes
:
АА*
dtype0
╘
!Adam/m/update_ip/recurrent_kernelVarHandleOp*
_output_shapes
: *2

debug_name$"Adam/m/update_ip/recurrent_kernel/*
dtype0*
shape:
АА*2
shared_name#!Adam/m/update_ip/recurrent_kernel
Щ
5Adam/m/update_ip/recurrent_kernel/Read/ReadVariableOpReadVariableOp!Adam/m/update_ip/recurrent_kernel* 
_output_shapes
:
АА*
dtype0
╢
Adam/v/update_ip/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/v/update_ip/kernel/*
dtype0*
shape:
АА*(
shared_nameAdam/v/update_ip/kernel
Е
+Adam/v/update_ip/kernel/Read/ReadVariableOpReadVariableOpAdam/v/update_ip/kernel* 
_output_shapes
:
АА*
dtype0
╢
Adam/m/update_ip/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/m/update_ip/kernel/*
dtype0*
shape:
АА*(
shared_nameAdam/m/update_ip/kernel
Е
+Adam/m/update_ip/kernel/Read/ReadVariableOpReadVariableOpAdam/m/update_ip/kernel* 
_output_shapes
:
АА*
dtype0
ж
current_learning_rateVarHandleOp*
_output_shapes
: *&

debug_namecurrent_learning_rate/*
dtype0*
shape: *&
shared_namecurrent_learning_rate
w
)current_learning_rate/Read/ReadVariableOpReadVariableOpcurrent_learning_rate*
_output_shapes
: *
dtype0
В
	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
П
dense_4/biasVarHandleOp*
_output_shapes
: *

debug_namedense_4/bias/*
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
Щ
dense_4/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_4/kernel/*
dtype0*
shape
:@*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:@*
dtype0
П
dense_3/biasVarHandleOp*
_output_shapes
: *

debug_namedense_3/bias/*
dtype0*
shape:@*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:@*
dtype0
Ъ
dense_3/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_3/kernel/*
dtype0*
shape:	А@*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	А@*
dtype0
Р
dense_2/biasVarHandleOp*
_output_shapes
: *

debug_namedense_2/bias/*
dtype0*
shape:А*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:А*
dtype0
Ы
dense_2/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_2/kernel/*
dtype0*
shape:
АА*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
АА*
dtype0
Р
dense_1/biasVarHandleOp*
_output_shapes
: *

debug_namedense_1/bias/*
dtype0*
shape:А*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:А*
dtype0
Ы
dense_1/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_1/kernel/*
dtype0*
shape:
АА*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
АА*
dtype0
К

dense/biasVarHandleOp*
_output_shapes
: *

debug_namedense/bias/*
dtype0*
shape:А*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:А*
dtype0
Х
dense/kernelVarHandleOp*
_output_shapes
: *

debug_namedense/kernel/*
dtype0*
shape:
АА*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
АА*
dtype0
▓
update_connection/biasVarHandleOp*
_output_shapes
: *'

debug_nameupdate_connection/bias/*
dtype0*
shape:	А*'
shared_nameupdate_connection/bias
В
*update_connection/bias/Read/ReadVariableOpReadVariableOpupdate_connection/bias*
_output_shapes
:	А*
dtype0
╫
"update_connection/recurrent_kernelVarHandleOp*
_output_shapes
: *3

debug_name%#update_connection/recurrent_kernel/*
dtype0*
shape:
АА*3
shared_name$"update_connection/recurrent_kernel
Ы
6update_connection/recurrent_kernel/Read/ReadVariableOpReadVariableOp"update_connection/recurrent_kernel* 
_output_shapes
:
АА*
dtype0
╣
update_connection/kernelVarHandleOp*
_output_shapes
: *)

debug_nameupdate_connection/kernel/*
dtype0*
shape:
АА*)
shared_nameupdate_connection/kernel
З
,update_connection/kernel/Read/ReadVariableOpReadVariableOpupdate_connection/kernel* 
_output_shapes
:
АА*
dtype0
Ъ
update_ip/biasVarHandleOp*
_output_shapes
: *

debug_nameupdate_ip/bias/*
dtype0*
shape:	А*
shared_nameupdate_ip/bias
r
"update_ip/bias/Read/ReadVariableOpReadVariableOpupdate_ip/bias*
_output_shapes
:	А*
dtype0
┐
update_ip/recurrent_kernelVarHandleOp*
_output_shapes
: *+

debug_nameupdate_ip/recurrent_kernel/*
dtype0*
shape:
АА*+
shared_nameupdate_ip/recurrent_kernel
Л
.update_ip/recurrent_kernel/Read/ReadVariableOpReadVariableOpupdate_ip/recurrent_kernel* 
_output_shapes
:
АА*
dtype0
б
update_ip/kernelVarHandleOp*
_output_shapes
: *!

debug_nameupdate_ip/kernel/*
dtype0*
shape:
АА*!
shared_nameupdate_ip/kernel
w
$update_ip/kernel/Read/ReadVariableOpReadVariableOpupdate_ip/kernel* 
_output_shapes
:
АА*
dtype0
Z
$serving_default_dst_connection_to_ipPlaceholder*
_output_shapes
:*
dtype0	
Z
$serving_default_dst_ip_to_connectionPlaceholder*
_output_shapes
:*
dtype0	
X
"serving_default_feature_connectionPlaceholder*
_output_shapes
:*
dtype0
T
serving_default_n_cPlaceholder*
_output_shapes
: *
dtype0	*
shape: 
T
serving_default_n_iPlaceholder*
_output_shapes
: *
dtype0	*
shape: 
Z
$serving_default_src_connection_to_ipPlaceholder*
_output_shapes
:*
dtype0	
Z
$serving_default_src_ip_to_connectionPlaceholder*
_output_shapes
:*
dtype0	
╠
StatefulPartitionedCallStatefulPartitionedCall$serving_default_dst_connection_to_ip$serving_default_dst_ip_to_connection"serving_default_feature_connectionserving_default_n_cserving_default_n_i$serving_default_src_connection_to_ip$serving_default_src_ip_to_connectiondense/kernel
dense/biasdense_1/kerneldense_1/biasupdate_ip/biasupdate_ip/kernelupdate_ip/recurrent_kernelupdate_connection/biasupdate_connection/kernel"update_connection/recurrent_kerneldense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/bias*"
Tin
2						*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *.
f)R'
%__inference_signature_wrapper_4847456

NoOpNoOp
Бс
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╗р
value░рBмр Bдр
п
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	ip_update
	connection_update

message_func1
message_func2
readout
	optimizer
call

signatures*
z
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15*
z
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15*
* 
░
 non_trainable_variables

!layers
"metrics
#layer_regularization_losses
$layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

%trace_0
&trace_1* 

'trace_0
(trace_1* 
* 
╙
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses
/_random_generator

kernel
recurrent_kernel
bias*
╙
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6_random_generator

kernel
recurrent_kernel
bias*
─
7layer-0
8layer_with_weights-0
8layer-1
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses*
─
?layer-0
@layer_with_weights-0
@layer-1
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses*
Я
Glayer_with_weights-0
Glayer-0
Hlayer-1
Ilayer_with_weights-1
Ilayer-2
Jlayer-3
Klayer_with_weights-2
Klayer-4
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses*
Й
R
_variables
S_iterations
T_current_learning_rate
U_index_dict
V
_momentums
W_velocities
X_update_step_xla*

Ytrace_0* 

Zserving_default* 
PJ
VARIABLE_VALUEupdate_ip/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEupdate_ip/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEupdate_ip/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEupdate_connection/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"update_connection/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEupdate_connection/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
JD
VARIABLE_VALUE
dense/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEdense_1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_2/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_2/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_3/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_3/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_4/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_4/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
	1

2
3
4*
Т
[0
\1
]2
^3
_4
`5
a6
b7
c8
d9
e10
f11
g12
h13
i14
j15
k16
l17
m18
n19
o20
p21
q22
r23
s24
t25
u26
v27
w28
x29
y30
z31
{32
|33
}34*
* 
* 
* 
* 
* 
* 

0
1
2*

0
1
2*
* 
Ц
~non_trainable_variables

layers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

Гtrace_0
Дtrace_1* 

Еtrace_0
Жtrace_1* 
* 

0
1
2*

0
1
2*
* 
Ш
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

Мtrace_0
Нtrace_1* 

Оtrace_0
Пtrace_1* 
* 
м
Р	variables
Сtrainable_variables
Тregularization_losses
У	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses
Ц_random_generator* 
м
Ч	variables
Шtrainable_variables
Щregularization_losses
Ъ	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses

kernel
bias*

0
1*

0
1*
* 
Ш
Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

вtrace_0
гtrace_1* 

дtrace_0
еtrace_1* 
м
ж	variables
зtrainable_variables
иregularization_losses
й	keras_api
к__call__
+л&call_and_return_all_conditional_losses
м_random_generator* 
м
н	variables
оtrainable_variables
пregularization_losses
░	keras_api
▒__call__
+▓&call_and_return_all_conditional_losses

kernel
bias*

0
1*

0
1*
* 
Ш
│non_trainable_variables
┤layers
╡metrics
 ╢layer_regularization_losses
╖layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*

╕trace_0
╣trace_1* 

║trace_0
╗trace_1* 
м
╝	variables
╜trainable_variables
╛regularization_losses
┐	keras_api
└__call__
+┴&call_and_return_all_conditional_losses

kernel
bias*
м
┬	variables
├trainable_variables
─regularization_losses
┼	keras_api
╞__call__
+╟&call_and_return_all_conditional_losses
╚_random_generator* 
м
╔	variables
╩trainable_variables
╦regularization_losses
╠	keras_api
═__call__
+╬&call_and_return_all_conditional_losses

kernel
bias*
м
╧	variables
╨trainable_variables
╤regularization_losses
╥	keras_api
╙__call__
+╘&call_and_return_all_conditional_losses
╒_random_generator* 
м
╓	variables
╫trainable_variables
╪regularization_losses
┘	keras_api
┌__call__
+█&call_and_return_all_conditional_losses

kernel
bias*
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
Ш
▄non_trainable_variables
▌layers
▐metrics
 ▀layer_regularization_losses
рlayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses*

сtrace_0
тtrace_1* 

уtrace_0
фtrace_1* 
в
S0
х1
ц2
ч3
ш4
щ5
ъ6
ы7
ь8
э9
ю10
я11
Ё12
ё13
Є14
є15
Ї16
ї17
Ў18
ў19
°20
∙21
·22
√23
№24
¤25
■26
 27
А28
Б29
В30
Г31
Д32*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEcurrent_learning_rate;optimizer/_current_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
К
х0
ч1
щ2
ы3
э4
я5
ё6
є7
ї8
ў9
∙10
√11
¤12
 13
Б14
Г15*
К
ц0
ш1
ъ2
ь3
ю4
Ё5
Є6
Ї7
Ў8
°9
·10
№11
■12
А13
В14
Д15*
* 
* 
* 
<
Е	variables
Ж	keras_api

Зtotal

Иcount*
M
Й	variables
К	keras_api

Лtotal

Мcount
Н
_fn_kwargs*
Л
О	variables
П	keras_api
Рtrue_positives
Сtrue_negatives
Тfalse_positives
Уfalse_negatives
Ф
thresholds*
`
Х	variables
Ц	keras_api
Ч
thresholds
Шtrue_positives
Щfalse_negatives*
`
Ъ	variables
Ы	keras_api
Ь
thresholds
Эtrue_positives
Юfalse_positives*
`
Я	variables
а	keras_api
б
thresholds
вtrue_positives
гfalse_negatives*
`
д	variables
е	keras_api
ж
thresholds
зtrue_positives
иfalse_positives*
`
й	variables
к	keras_api
л
thresholds
мtrue_positives
нfalse_negatives*
`
о	variables
п	keras_api
░
thresholds
▒true_positives
▓false_positives*
`
│	variables
┤	keras_api
╡
thresholds
╢true_positives
╖false_negatives*
`
╕	variables
╣	keras_api
║
thresholds
╗true_positives
╝false_positives*
`
╜	variables
╛	keras_api
┐
thresholds
└true_positives
┴false_negatives*
`
┬	variables
├	keras_api
─
thresholds
┼true_positives
╞false_positives*
`
╟	variables
╚	keras_api
╔
thresholds
╩true_positives
╦false_negatives*
`
╠	variables
═	keras_api
╬
thresholds
╧true_positives
╨false_positives*
`
╤	variables
╥	keras_api
╙
thresholds
╘true_positives
╒false_negatives*
`
╓	variables
╫	keras_api
╪
thresholds
┘true_positives
┌false_positives*
`
█	variables
▄	keras_api
▌
thresholds
▐true_positives
▀false_negatives*
`
р	variables
с	keras_api
т
thresholds
уtrue_positives
фfalse_positives*
`
х	variables
ц	keras_api
ч
thresholds
шtrue_positives
щfalse_negatives*
`
ъ	variables
ы	keras_api
ь
thresholds
эtrue_positives
юfalse_positives*
`
я	variables
Ё	keras_api
ё
thresholds
Єtrue_positives
єfalse_negatives*
`
Ї	variables
ї	keras_api
Ў
thresholds
ўtrue_positives
°false_positives*
`
∙	variables
·	keras_api
√
thresholds
№true_positives
¤false_negatives*
`
■	variables
 	keras_api
А
thresholds
Бtrue_positives
Вfalse_positives*
`
Г	variables
Д	keras_api
Е
thresholds
Жtrue_positives
Зfalse_negatives*
`
И	variables
Й	keras_api
К
thresholds
Лtrue_positives
Мfalse_positives*
`
Н	variables
О	keras_api
П
thresholds
Рtrue_positives
Сfalse_negatives*
`
Т	variables
У	keras_api
Ф
thresholds
Хtrue_positives
Цfalse_positives*
`
Ч	variables
Ш	keras_api
Щ
thresholds
Ъtrue_positives
Ыfalse_negatives*
`
Ь	variables
Э	keras_api
Ю
thresholds
Яtrue_positives
аfalse_positives*
`
б	variables
в	keras_api
г
thresholds
дtrue_positives
еfalse_negatives*
`
ж	variables
з	keras_api
и
thresholds
йtrue_positives
кfalse_positives*
С
л	variables
м	keras_api
н
init_shape
оtrue_positives
пfalse_positives
░false_negatives
▒weights_intermediate*
С
▓	variables
│	keras_api
┤
init_shape
╡true_positives
╢false_positives
╖false_negatives
╕weights_intermediate*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Ь
╣non_trainable_variables
║layers
╗metrics
 ╝layer_regularization_losses
╜layer_metrics
Р	variables
Сtrainable_variables
Тregularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses* 

╛trace_0
┐trace_1* 

└trace_0
┴trace_1* 
* 

0
1*

0
1*
* 
Ю
┬non_trainable_variables
├layers
─metrics
 ┼layer_regularization_losses
╞layer_metrics
Ч	variables
Шtrainable_variables
Щregularization_losses
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses*

╟trace_0* 

╚trace_0* 
* 

70
81*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Ь
╔non_trainable_variables
╩layers
╦metrics
 ╠layer_regularization_losses
═layer_metrics
ж	variables
зtrainable_variables
иregularization_losses
к__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses* 

╬trace_0
╧trace_1* 

╨trace_0
╤trace_1* 
* 

0
1*

0
1*
* 
Ю
╥non_trainable_variables
╙layers
╘metrics
 ╒layer_regularization_losses
╓layer_metrics
н	variables
оtrainable_variables
пregularization_losses
▒__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses*

╫trace_0* 

╪trace_0* 
* 

?0
@1*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
Ю
┘non_trainable_variables
┌layers
█metrics
 ▄layer_regularization_losses
▌layer_metrics
╝	variables
╜trainable_variables
╛regularization_losses
└__call__
+┴&call_and_return_all_conditional_losses
'┴"call_and_return_conditional_losses*

▐trace_0* 

▀trace_0* 
* 
* 
* 
Ь
рnon_trainable_variables
сlayers
тmetrics
 уlayer_regularization_losses
фlayer_metrics
┬	variables
├trainable_variables
─regularization_losses
╞__call__
+╟&call_and_return_all_conditional_losses
'╟"call_and_return_conditional_losses* 

хtrace_0
цtrace_1* 

чtrace_0
шtrace_1* 
* 

0
1*

0
1*
* 
Ю
щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
╔	variables
╩trainable_variables
╦regularization_losses
═__call__
+╬&call_and_return_all_conditional_losses
'╬"call_and_return_conditional_losses*

юtrace_0* 

яtrace_0* 
* 
* 
* 
Ь
Ёnon_trainable_variables
ёlayers
Єmetrics
 єlayer_regularization_losses
Їlayer_metrics
╧	variables
╨trainable_variables
╤regularization_losses
╙__call__
+╘&call_and_return_all_conditional_losses
'╘"call_and_return_conditional_losses* 

їtrace_0
Ўtrace_1* 

ўtrace_0
°trace_1* 
* 

0
1*

0
1*
* 
Ю
∙non_trainable_variables
·layers
√metrics
 №layer_regularization_losses
¤layer_metrics
╓	variables
╫trainable_variables
╪regularization_losses
┌__call__
+█&call_and_return_all_conditional_losses
'█"call_and_return_conditional_losses*

■trace_0* 

 trace_0* 
* 
'
G0
H1
I2
J3
K4*
* 
* 
* 
* 
* 
* 
* 
b\
VARIABLE_VALUEAdam/m/update_ip/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/update_ip/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE!Adam/m/update_ip/recurrent_kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE!Adam/v/update_ip/recurrent_kernel1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/update_ip/bias1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/update_ip/bias1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/update_connection/kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/update_connection/kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE)Adam/m/update_connection/recurrent_kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE)Adam/v/update_connection/recurrent_kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/m/update_connection/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/v/update_connection/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/dense/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/dense/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_1/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_1/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_1/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_1/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_2/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_2/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_2/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_2/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_3/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_3/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_3/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_3/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_4/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_4/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_4/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_4/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*

З0
И1*

Е	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Л0
М1*

Й	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
$
Р0
С1
Т2
У3*

О	variables*
hb
VARIABLE_VALUEtrue_positives_32=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_positives_17>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_negatives_17>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
* 

Ш0
Щ1*

Х	variables*
* 
hb
VARIABLE_VALUEtrue_positives_31=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_negatives_16>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*

Э0
Ю1*

Ъ	variables*
* 
hb
VARIABLE_VALUEtrue_positives_30=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_positives_16>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUE*

в0
г1*

Я	variables*
* 
hb
VARIABLE_VALUEtrue_positives_29=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_negatives_15>keras_api/metrics/5/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*

з0
и1*

д	variables*
* 
hb
VARIABLE_VALUEtrue_positives_28=keras_api/metrics/6/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_positives_15>keras_api/metrics/6/false_positives/.ATTRIBUTES/VARIABLE_VALUE*

м0
н1*

й	variables*
* 
hb
VARIABLE_VALUEtrue_positives_27=keras_api/metrics/7/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_negatives_14>keras_api/metrics/7/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*

▒0
▓1*

о	variables*
* 
hb
VARIABLE_VALUEtrue_positives_26=keras_api/metrics/8/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_positives_14>keras_api/metrics/8/false_positives/.ATTRIBUTES/VARIABLE_VALUE*

╢0
╖1*

│	variables*
* 
hb
VARIABLE_VALUEtrue_positives_25=keras_api/metrics/9/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_negatives_13>keras_api/metrics/9/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*

╗0
╝1*

╕	variables*
* 
ic
VARIABLE_VALUEtrue_positives_24>keras_api/metrics/10/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEfalse_positives_13?keras_api/metrics/10/false_positives/.ATTRIBUTES/VARIABLE_VALUE*

└0
┴1*

╜	variables*
* 
ic
VARIABLE_VALUEtrue_positives_23>keras_api/metrics/11/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEfalse_negatives_12?keras_api/metrics/11/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*

┼0
╞1*

┬	variables*
* 
ic
VARIABLE_VALUEtrue_positives_22>keras_api/metrics/12/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEfalse_positives_12?keras_api/metrics/12/false_positives/.ATTRIBUTES/VARIABLE_VALUE*

╩0
╦1*

╟	variables*
* 
ic
VARIABLE_VALUEtrue_positives_21>keras_api/metrics/13/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEfalse_negatives_11?keras_api/metrics/13/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*

╧0
╨1*

╠	variables*
* 
ic
VARIABLE_VALUEtrue_positives_20>keras_api/metrics/14/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEfalse_positives_11?keras_api/metrics/14/false_positives/.ATTRIBUTES/VARIABLE_VALUE*

╘0
╒1*

╤	variables*
* 
ic
VARIABLE_VALUEtrue_positives_19>keras_api/metrics/15/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEfalse_negatives_10?keras_api/metrics/15/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*

┘0
┌1*

╓	variables*
* 
ic
VARIABLE_VALUEtrue_positives_18>keras_api/metrics/16/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEfalse_positives_10?keras_api/metrics/16/false_positives/.ATTRIBUTES/VARIABLE_VALUE*

▐0
▀1*

█	variables*
* 
ic
VARIABLE_VALUEtrue_positives_17>keras_api/metrics/17/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_negatives_9?keras_api/metrics/17/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*

у0
ф1*

р	variables*
* 
ic
VARIABLE_VALUEtrue_positives_16>keras_api/metrics/18/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_positives_9?keras_api/metrics/18/false_positives/.ATTRIBUTES/VARIABLE_VALUE*

ш0
щ1*

х	variables*
* 
ic
VARIABLE_VALUEtrue_positives_15>keras_api/metrics/19/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_negatives_8?keras_api/metrics/19/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*

э0
ю1*

ъ	variables*
* 
ic
VARIABLE_VALUEtrue_positives_14>keras_api/metrics/20/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_positives_8?keras_api/metrics/20/false_positives/.ATTRIBUTES/VARIABLE_VALUE*

Є0
є1*

я	variables*
* 
ic
VARIABLE_VALUEtrue_positives_13>keras_api/metrics/21/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_negatives_7?keras_api/metrics/21/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*

ў0
°1*

Ї	variables*
* 
ic
VARIABLE_VALUEtrue_positives_12>keras_api/metrics/22/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_positives_7?keras_api/metrics/22/false_positives/.ATTRIBUTES/VARIABLE_VALUE*

№0
¤1*

∙	variables*
* 
ic
VARIABLE_VALUEtrue_positives_11>keras_api/metrics/23/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_negatives_6?keras_api/metrics/23/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*

Б0
В1*

■	variables*
* 
ic
VARIABLE_VALUEtrue_positives_10>keras_api/metrics/24/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_positives_6?keras_api/metrics/24/false_positives/.ATTRIBUTES/VARIABLE_VALUE*

Ж0
З1*

Г	variables*
* 
hb
VARIABLE_VALUEtrue_positives_9>keras_api/metrics/25/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_negatives_5?keras_api/metrics/25/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*

Л0
М1*

И	variables*
* 
hb
VARIABLE_VALUEtrue_positives_8>keras_api/metrics/26/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_positives_5?keras_api/metrics/26/false_positives/.ATTRIBUTES/VARIABLE_VALUE*

Р0
С1*

Н	variables*
* 
hb
VARIABLE_VALUEtrue_positives_7>keras_api/metrics/27/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_negatives_4?keras_api/metrics/27/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*

Х0
Ц1*

Т	variables*
* 
hb
VARIABLE_VALUEtrue_positives_6>keras_api/metrics/28/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_positives_4?keras_api/metrics/28/false_positives/.ATTRIBUTES/VARIABLE_VALUE*

Ъ0
Ы1*

Ч	variables*
* 
hb
VARIABLE_VALUEtrue_positives_5>keras_api/metrics/29/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_negatives_3?keras_api/metrics/29/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*

Я0
а1*

Ь	variables*
* 
hb
VARIABLE_VALUEtrue_positives_4>keras_api/metrics/30/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_positives_3?keras_api/metrics/30/false_positives/.ATTRIBUTES/VARIABLE_VALUE*

д0
е1*

б	variables*
* 
hb
VARIABLE_VALUEtrue_positives_3>keras_api/metrics/31/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_negatives_2?keras_api/metrics/31/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*

й0
к1*

ж	variables*
* 
hb
VARIABLE_VALUEtrue_positives_2>keras_api/metrics/32/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_positives_2?keras_api/metrics/32/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
$
о0
п1
░2
▒3*

л	variables*
* 
hb
VARIABLE_VALUEtrue_positives_1>keras_api/metrics/33/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_positives_1?keras_api/metrics/33/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEfalse_negatives_1?keras_api/metrics/33/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEweights_intermediate_1Dkeras_api/metrics/33/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUE*
$
╡0
╢1
╖2
╕3*

▓	variables*
* 
f`
VARIABLE_VALUEtrue_positives>keras_api/metrics/34/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEfalse_positives?keras_api/metrics/34/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEfalse_negatives?keras_api/metrics/34/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEweights_intermediateDkeras_api/metrics/34/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
П
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameupdate_ip/kernelupdate_ip/recurrent_kernelupdate_ip/biasupdate_connection/kernel"update_connection/recurrent_kernelupdate_connection/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/bias	iterationcurrent_learning_rateAdam/m/update_ip/kernelAdam/v/update_ip/kernel!Adam/m/update_ip/recurrent_kernel!Adam/v/update_ip/recurrent_kernelAdam/m/update_ip/biasAdam/v/update_ip/biasAdam/m/update_connection/kernelAdam/v/update_connection/kernel)Adam/m/update_connection/recurrent_kernel)Adam/v/update_connection/recurrent_kernelAdam/m/update_connection/biasAdam/v/update_connection/biasAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biasAdam/m/dense_2/kernelAdam/v/dense_2/kernelAdam/m/dense_2/biasAdam/v/dense_2/biasAdam/m/dense_3/kernelAdam/v/dense_3/kernelAdam/m/dense_3/biasAdam/v/dense_3/biasAdam/m/dense_4/kernelAdam/v/dense_4/kernelAdam/m/dense_4/biasAdam/v/dense_4/biastotal_1count_1totalcounttrue_positives_32true_negativesfalse_positives_17false_negatives_17true_positives_31false_negatives_16true_positives_30false_positives_16true_positives_29false_negatives_15true_positives_28false_positives_15true_positives_27false_negatives_14true_positives_26false_positives_14true_positives_25false_negatives_13true_positives_24false_positives_13true_positives_23false_negatives_12true_positives_22false_positives_12true_positives_21false_negatives_11true_positives_20false_positives_11true_positives_19false_negatives_10true_positives_18false_positives_10true_positives_17false_negatives_9true_positives_16false_positives_9true_positives_15false_negatives_8true_positives_14false_positives_8true_positives_13false_negatives_7true_positives_12false_positives_7true_positives_11false_negatives_6true_positives_10false_positives_6true_positives_9false_negatives_5true_positives_8false_positives_5true_positives_7false_negatives_4true_positives_6false_positives_4true_positives_5false_negatives_3true_positives_4false_positives_3true_positives_3false_negatives_2true_positives_2false_positives_2true_positives_1false_positives_1false_negatives_1weights_intermediate_1true_positivesfalse_positivesfalse_negativesweights_intermediateConst*О
TinЖ
Г2А*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__traced_save_4848660
Й
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameupdate_ip/kernelupdate_ip/recurrent_kernelupdate_ip/biasupdate_connection/kernel"update_connection/recurrent_kernelupdate_connection/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/bias	iterationcurrent_learning_rateAdam/m/update_ip/kernelAdam/v/update_ip/kernel!Adam/m/update_ip/recurrent_kernel!Adam/v/update_ip/recurrent_kernelAdam/m/update_ip/biasAdam/v/update_ip/biasAdam/m/update_connection/kernelAdam/v/update_connection/kernel)Adam/m/update_connection/recurrent_kernel)Adam/v/update_connection/recurrent_kernelAdam/m/update_connection/biasAdam/v/update_connection/biasAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biasAdam/m/dense_2/kernelAdam/v/dense_2/kernelAdam/m/dense_2/biasAdam/v/dense_2/biasAdam/m/dense_3/kernelAdam/v/dense_3/kernelAdam/m/dense_3/biasAdam/v/dense_3/biasAdam/m/dense_4/kernelAdam/v/dense_4/kernelAdam/m/dense_4/biasAdam/v/dense_4/biastotal_1count_1totalcounttrue_positives_32true_negativesfalse_positives_17false_negatives_17true_positives_31false_negatives_16true_positives_30false_positives_16true_positives_29false_negatives_15true_positives_28false_positives_15true_positives_27false_negatives_14true_positives_26false_positives_14true_positives_25false_negatives_13true_positives_24false_positives_13true_positives_23false_negatives_12true_positives_22false_positives_12true_positives_21false_negatives_11true_positives_20false_positives_11true_positives_19false_negatives_10true_positives_18false_positives_10true_positives_17false_negatives_9true_positives_16false_positives_9true_positives_15false_negatives_8true_positives_14false_positives_8true_positives_13false_negatives_7true_positives_12false_positives_7true_positives_11false_negatives_6true_positives_10false_positives_6true_positives_9false_negatives_5true_positives_8false_positives_5true_positives_7false_negatives_4true_positives_6false_positives_4true_positives_5false_negatives_3true_positives_4false_positives_3true_positives_3false_negatives_2true_positives_2false_positives_2true_positives_1false_positives_1false_negatives_1weights_intermediate_1true_positivesfalse_positivesfalse_negativesweights_intermediate*М
TinД
Б2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference__traced_restore_4849047╧о1
Ё
Ц
)__inference_dense_4_layer_call_fn_4847865

inputs
unknown:@
	unknown_0:
identityИвStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_4845649o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	4847861:'#
!
_user_specified_name	4847859:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
є
┌
F__inference_update_ip_layer_call_and_return_conditional_losses_4845902

inputs

states*
readvariableop_resource:	А2
matmul_readvariableop_resource:
АА4
 matmul_1_readvariableop_resource:
АА
identity

identity_1ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	А*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:А:А*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         АZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         АZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"А   А       \
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╔
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         АN
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         Аc
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         АR
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         А^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         АZ
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         АJ
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:         АT
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:         АJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         АR
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:         АW
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         АY
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         А[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         Аe
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         А:         А: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:PL
(
_output_shapes
:         А
 
_user_specified_namestates:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
б

e
F__inference_dropout_2_layer_call_and_return_conditional_losses_4845608

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╧
d
+__inference_dropout_3_layer_call_fn_4847834

inputs
identityИвStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_4845637o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
█┐
ФM
#__inference__traced_restore_4849047
file_prefix5
!assignvariableop_update_ip_kernel:
ААA
-assignvariableop_1_update_ip_recurrent_kernel:
АА4
!assignvariableop_2_update_ip_bias:	А?
+assignvariableop_3_update_connection_kernel:
ААI
5assignvariableop_4_update_connection_recurrent_kernel:
АА<
)assignvariableop_5_update_connection_bias:	А3
assignvariableop_6_dense_kernel:
АА,
assignvariableop_7_dense_bias:	А5
!assignvariableop_8_dense_1_kernel:
АА.
assignvariableop_9_dense_1_bias:	А6
"assignvariableop_10_dense_2_kernel:
АА/
 assignvariableop_11_dense_2_bias:	А5
"assignvariableop_12_dense_3_kernel:	А@.
 assignvariableop_13_dense_3_bias:@4
"assignvariableop_14_dense_4_kernel:@.
 assignvariableop_15_dense_4_bias:'
assignvariableop_16_iteration:	 3
)assignvariableop_17_current_learning_rate: ?
+assignvariableop_18_adam_m_update_ip_kernel:
АА?
+assignvariableop_19_adam_v_update_ip_kernel:
ААI
5assignvariableop_20_adam_m_update_ip_recurrent_kernel:
ААI
5assignvariableop_21_adam_v_update_ip_recurrent_kernel:
АА<
)assignvariableop_22_adam_m_update_ip_bias:	А<
)assignvariableop_23_adam_v_update_ip_bias:	АG
3assignvariableop_24_adam_m_update_connection_kernel:
ААG
3assignvariableop_25_adam_v_update_connection_kernel:
ААQ
=assignvariableop_26_adam_m_update_connection_recurrent_kernel:
ААQ
=assignvariableop_27_adam_v_update_connection_recurrent_kernel:
ААD
1assignvariableop_28_adam_m_update_connection_bias:	АD
1assignvariableop_29_adam_v_update_connection_bias:	А;
'assignvariableop_30_adam_m_dense_kernel:
АА;
'assignvariableop_31_adam_v_dense_kernel:
АА4
%assignvariableop_32_adam_m_dense_bias:	А4
%assignvariableop_33_adam_v_dense_bias:	А=
)assignvariableop_34_adam_m_dense_1_kernel:
АА=
)assignvariableop_35_adam_v_dense_1_kernel:
АА6
'assignvariableop_36_adam_m_dense_1_bias:	А6
'assignvariableop_37_adam_v_dense_1_bias:	А=
)assignvariableop_38_adam_m_dense_2_kernel:
АА=
)assignvariableop_39_adam_v_dense_2_kernel:
АА6
'assignvariableop_40_adam_m_dense_2_bias:	А6
'assignvariableop_41_adam_v_dense_2_bias:	А<
)assignvariableop_42_adam_m_dense_3_kernel:	А@<
)assignvariableop_43_adam_v_dense_3_kernel:	А@5
'assignvariableop_44_adam_m_dense_3_bias:@5
'assignvariableop_45_adam_v_dense_3_bias:@;
)assignvariableop_46_adam_m_dense_4_kernel:@;
)assignvariableop_47_adam_v_dense_4_kernel:@5
'assignvariableop_48_adam_m_dense_4_bias:5
'assignvariableop_49_adam_v_dense_4_bias:%
assignvariableop_50_total_1: %
assignvariableop_51_count_1: #
assignvariableop_52_total: #
assignvariableop_53_count: 4
%assignvariableop_54_true_positives_32:	╚1
"assignvariableop_55_true_negatives:	╚5
&assignvariableop_56_false_positives_17:	╚5
&assignvariableop_57_false_negatives_17:	╚3
%assignvariableop_58_true_positives_31:4
&assignvariableop_59_false_negatives_16:3
%assignvariableop_60_true_positives_30:4
&assignvariableop_61_false_positives_16:3
%assignvariableop_62_true_positives_29:4
&assignvariableop_63_false_negatives_15:3
%assignvariableop_64_true_positives_28:4
&assignvariableop_65_false_positives_15:3
%assignvariableop_66_true_positives_27:4
&assignvariableop_67_false_negatives_14:3
%assignvariableop_68_true_positives_26:4
&assignvariableop_69_false_positives_14:3
%assignvariableop_70_true_positives_25:4
&assignvariableop_71_false_negatives_13:3
%assignvariableop_72_true_positives_24:4
&assignvariableop_73_false_positives_13:3
%assignvariableop_74_true_positives_23:4
&assignvariableop_75_false_negatives_12:3
%assignvariableop_76_true_positives_22:4
&assignvariableop_77_false_positives_12:3
%assignvariableop_78_true_positives_21:4
&assignvariableop_79_false_negatives_11:3
%assignvariableop_80_true_positives_20:4
&assignvariableop_81_false_positives_11:3
%assignvariableop_82_true_positives_19:4
&assignvariableop_83_false_negatives_10:3
%assignvariableop_84_true_positives_18:4
&assignvariableop_85_false_positives_10:3
%assignvariableop_86_true_positives_17:3
%assignvariableop_87_false_negatives_9:3
%assignvariableop_88_true_positives_16:3
%assignvariableop_89_false_positives_9:3
%assignvariableop_90_true_positives_15:3
%assignvariableop_91_false_negatives_8:3
%assignvariableop_92_true_positives_14:3
%assignvariableop_93_false_positives_8:3
%assignvariableop_94_true_positives_13:3
%assignvariableop_95_false_negatives_7:3
%assignvariableop_96_true_positives_12:3
%assignvariableop_97_false_positives_7:3
%assignvariableop_98_true_positives_11:3
%assignvariableop_99_false_negatives_6:4
&assignvariableop_100_true_positives_10:4
&assignvariableop_101_false_positives_6:3
%assignvariableop_102_true_positives_9:4
&assignvariableop_103_false_negatives_5:3
%assignvariableop_104_true_positives_8:4
&assignvariableop_105_false_positives_5:3
%assignvariableop_106_true_positives_7:4
&assignvariableop_107_false_negatives_4:3
%assignvariableop_108_true_positives_6:4
&assignvariableop_109_false_positives_4:3
%assignvariableop_110_true_positives_5:4
&assignvariableop_111_false_negatives_3:3
%assignvariableop_112_true_positives_4:4
&assignvariableop_113_false_positives_3:3
%assignvariableop_114_true_positives_3:4
&assignvariableop_115_false_negatives_2:3
%assignvariableop_116_true_positives_2:4
&assignvariableop_117_false_positives_2:3
%assignvariableop_118_true_positives_1:4
&assignvariableop_119_false_positives_1:4
&assignvariableop_120_false_negatives_1:9
+assignvariableop_121_weights_intermediate_1:1
#assignvariableop_122_true_positives:2
$assignvariableop_123_false_positives:2
$assignvariableop_124_false_negatives:7
)assignvariableop_125_weights_intermediate:
identity_127ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_100вAssignVariableOp_101вAssignVariableOp_102вAssignVariableOp_103вAssignVariableOp_104вAssignVariableOp_105вAssignVariableOp_106вAssignVariableOp_107вAssignVariableOp_108вAssignVariableOp_109вAssignVariableOp_11вAssignVariableOp_110вAssignVariableOp_111вAssignVariableOp_112вAssignVariableOp_113вAssignVariableOp_114вAssignVariableOp_115вAssignVariableOp_116вAssignVariableOp_117вAssignVariableOp_118вAssignVariableOp_119вAssignVariableOp_12вAssignVariableOp_120вAssignVariableOp_121вAssignVariableOp_122вAssignVariableOp_123вAssignVariableOp_124вAssignVariableOp_125вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_56вAssignVariableOp_57вAssignVariableOp_58вAssignVariableOp_59вAssignVariableOp_6вAssignVariableOp_60вAssignVariableOp_61вAssignVariableOp_62вAssignVariableOp_63вAssignVariableOp_64вAssignVariableOp_65вAssignVariableOp_66вAssignVariableOp_67вAssignVariableOp_68вAssignVariableOp_69вAssignVariableOp_7вAssignVariableOp_70вAssignVariableOp_71вAssignVariableOp_72вAssignVariableOp_73вAssignVariableOp_74вAssignVariableOp_75вAssignVariableOp_76вAssignVariableOp_77вAssignVariableOp_78вAssignVariableOp_79вAssignVariableOp_8вAssignVariableOp_80вAssignVariableOp_81вAssignVariableOp_82вAssignVariableOp_83вAssignVariableOp_84вAssignVariableOp_85вAssignVariableOp_86вAssignVariableOp_87вAssignVariableOp_88вAssignVariableOp_89вAssignVariableOp_9вAssignVariableOp_90вAssignVariableOp_91вAssignVariableOp_92вAssignVariableOp_93вAssignVariableOp_94вAssignVariableOp_95вAssignVariableOp_96вAssignVariableOp_97вAssignVariableOp_98вAssignVariableOp_99я9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Х9
valueЛ9BИ9B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_current_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/5/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/6/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/6/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/7/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/7/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/8/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/8/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/9/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/9/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/10/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/10/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/11/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/11/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/12/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/12/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/13/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/13/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/14/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/14/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/15/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/15/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/16/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/16/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/17/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/17/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/18/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/18/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/19/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/19/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/20/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/20/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/21/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/21/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/22/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/22/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/23/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/23/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/24/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/24/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/25/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/25/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/26/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/26/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/27/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/27/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/28/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/28/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/29/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/29/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/30/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/30/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/31/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/31/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/32/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/32/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/33/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/33/false_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/33/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBDkeras_api/metrics/33/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/34/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/34/false_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/34/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBDkeras_api/metrics/34/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHё
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*У
valueЙBЖB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Я
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Т
_output_shapes 
№:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*П
dtypesД
Б2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOpAssignVariableOp!assignvariableop_update_ip_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_1AssignVariableOp-assignvariableop_1_update_ip_recurrent_kernelIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_2AssignVariableOp!assignvariableop_2_update_ip_biasIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_3AssignVariableOp+assignvariableop_3_update_connection_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_4AssignVariableOp5assignvariableop_4_update_connection_recurrent_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_5AssignVariableOp)assignvariableop_5_update_connection_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_1_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_1_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_2_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_2_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_3_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_3_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_4_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_4_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:╢
AssignVariableOp_16AssignVariableOpassignvariableop_16_iterationIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_17AssignVariableOp)assignvariableop_17_current_learning_rateIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_18AssignVariableOp+assignvariableop_18_adam_m_update_ip_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_v_update_ip_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_20AssignVariableOp5assignvariableop_20_adam_m_update_ip_recurrent_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_21AssignVariableOp5assignvariableop_21_adam_v_update_ip_recurrent_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_m_update_ip_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_v_update_ip_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_24AssignVariableOp3assignvariableop_24_adam_m_update_connection_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_25AssignVariableOp3assignvariableop_25_adam_v_update_connection_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:╓
AssignVariableOp_26AssignVariableOp=assignvariableop_26_adam_m_update_connection_recurrent_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:╓
AssignVariableOp_27AssignVariableOp=assignvariableop_27_adam_v_update_connection_recurrent_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:╩
AssignVariableOp_28AssignVariableOp1assignvariableop_28_adam_m_update_connection_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:╩
AssignVariableOp_29AssignVariableOp1assignvariableop_29_adam_v_update_connection_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_m_dense_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_v_dense_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_m_dense_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_33AssignVariableOp%assignvariableop_33_adam_v_dense_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_m_dense_1_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_v_dense_1_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_m_dense_1_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adam_v_dense_1_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_m_dense_2_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_v_dense_2_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_m_dense_2_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_41AssignVariableOp'assignvariableop_41_adam_v_dense_2_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_m_dense_3_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_v_dense_3_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_m_dense_3_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_45AssignVariableOp'assignvariableop_45_adam_v_dense_3_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_m_dense_4_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_v_dense_4_kernelIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_48AssignVariableOp'assignvariableop_48_adam_m_dense_4_biasIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_49AssignVariableOp'assignvariableop_49_adam_v_dense_4_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_50AssignVariableOpassignvariableop_50_total_1Identity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_51AssignVariableOpassignvariableop_51_count_1Identity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_52AssignVariableOpassignvariableop_52_totalIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_53AssignVariableOpassignvariableop_53_countIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_54AssignVariableOp%assignvariableop_54_true_positives_32Identity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:╗
AssignVariableOp_55AssignVariableOp"assignvariableop_55_true_negativesIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_56AssignVariableOp&assignvariableop_56_false_positives_17Identity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_57AssignVariableOp&assignvariableop_57_false_negatives_17Identity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_58AssignVariableOp%assignvariableop_58_true_positives_31Identity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_59AssignVariableOp&assignvariableop_59_false_negatives_16Identity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_60AssignVariableOp%assignvariableop_60_true_positives_30Identity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_61AssignVariableOp&assignvariableop_61_false_positives_16Identity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_62AssignVariableOp%assignvariableop_62_true_positives_29Identity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_63AssignVariableOp&assignvariableop_63_false_negatives_15Identity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_64AssignVariableOp%assignvariableop_64_true_positives_28Identity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_65AssignVariableOp&assignvariableop_65_false_positives_15Identity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_66AssignVariableOp%assignvariableop_66_true_positives_27Identity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_67AssignVariableOp&assignvariableop_67_false_negatives_14Identity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_68AssignVariableOp%assignvariableop_68_true_positives_26Identity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_69AssignVariableOp&assignvariableop_69_false_positives_14Identity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_70AssignVariableOp%assignvariableop_70_true_positives_25Identity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_71AssignVariableOp&assignvariableop_71_false_negatives_13Identity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_72AssignVariableOp%assignvariableop_72_true_positives_24Identity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_73AssignVariableOp&assignvariableop_73_false_positives_13Identity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_74AssignVariableOp%assignvariableop_74_true_positives_23Identity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_75AssignVariableOp&assignvariableop_75_false_negatives_12Identity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_76AssignVariableOp%assignvariableop_76_true_positives_22Identity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_77AssignVariableOp&assignvariableop_77_false_positives_12Identity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_78AssignVariableOp%assignvariableop_78_true_positives_21Identity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_79AssignVariableOp&assignvariableop_79_false_negatives_11Identity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_80AssignVariableOp%assignvariableop_80_true_positives_20Identity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_81AssignVariableOp&assignvariableop_81_false_positives_11Identity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_82AssignVariableOp%assignvariableop_82_true_positives_19Identity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_83AssignVariableOp&assignvariableop_83_false_negatives_10Identity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_84AssignVariableOp%assignvariableop_84_true_positives_18Identity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_85AssignVariableOp&assignvariableop_85_false_positives_10Identity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_86AssignVariableOp%assignvariableop_86_true_positives_17Identity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_87AssignVariableOp%assignvariableop_87_false_negatives_9Identity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_88AssignVariableOp%assignvariableop_88_true_positives_16Identity_88:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_89AssignVariableOp%assignvariableop_89_false_positives_9Identity_89:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_90AssignVariableOp%assignvariableop_90_true_positives_15Identity_90:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_91AssignVariableOp%assignvariableop_91_false_negatives_8Identity_91:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_92AssignVariableOp%assignvariableop_92_true_positives_14Identity_92:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_93AssignVariableOp%assignvariableop_93_false_positives_8Identity_93:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_94AssignVariableOp%assignvariableop_94_true_positives_13Identity_94:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_95AssignVariableOp%assignvariableop_95_false_negatives_7Identity_95:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_96AssignVariableOp%assignvariableop_96_true_positives_12Identity_96:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_97AssignVariableOp%assignvariableop_97_false_positives_7Identity_97:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_98AssignVariableOp%assignvariableop_98_true_positives_11Identity_98:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_99AssignVariableOp%assignvariableop_99_false_negatives_6Identity_99:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_100AssignVariableOp&assignvariableop_100_true_positives_10Identity_100:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_101AssignVariableOp&assignvariableop_101_false_positives_6Identity_101:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_102AssignVariableOp%assignvariableop_102_true_positives_9Identity_102:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_103AssignVariableOp&assignvariableop_103_false_negatives_5Identity_103:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_104AssignVariableOp%assignvariableop_104_true_positives_8Identity_104:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_105AssignVariableOp&assignvariableop_105_false_positives_5Identity_105:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_106AssignVariableOp%assignvariableop_106_true_positives_7Identity_106:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_107AssignVariableOp&assignvariableop_107_false_negatives_4Identity_107:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_108AssignVariableOp%assignvariableop_108_true_positives_6Identity_108:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_109AssignVariableOp&assignvariableop_109_false_positives_4Identity_109:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_110AssignVariableOp%assignvariableop_110_true_positives_5Identity_110:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_111AssignVariableOp&assignvariableop_111_false_negatives_3Identity_111:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_112AssignVariableOp%assignvariableop_112_true_positives_4Identity_112:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_113AssignVariableOp&assignvariableop_113_false_positives_3Identity_113:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_114AssignVariableOp%assignvariableop_114_true_positives_3Identity_114:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_115AssignVariableOp&assignvariableop_115_false_negatives_2Identity_115:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_116AssignVariableOp%assignvariableop_116_true_positives_2Identity_116:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_117AssignVariableOp&assignvariableop_117_false_positives_2Identity_117:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_118AssignVariableOp%assignvariableop_118_true_positives_1Identity_118:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_119AssignVariableOp&assignvariableop_119_false_positives_1Identity_119:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:┴
AssignVariableOp_120AssignVariableOp&assignvariableop_120_false_negatives_1Identity_120:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:╞
AssignVariableOp_121AssignVariableOp+assignvariableop_121_weights_intermediate_1Identity_121:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_122AssignVariableOp#assignvariableop_122_true_positivesIdentity_122:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_123AssignVariableOp$assignvariableop_123_false_positivesIdentity_123:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:┐
AssignVariableOp_124AssignVariableOp$assignvariableop_124_false_negativesIdentity_124:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:─
AssignVariableOp_125AssignVariableOp)assignvariableop_125_weights_intermediateIdentity_125:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ╛
Identity_126Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_127IdentityIdentity_126:output:0^NoOp_1*
T0*
_output_shapes
: Ж
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
_output_shapes
 "%
identity_127Identity_127:output:0*(
_construction_contextkEagerRuntime*У
_input_shapesБ
■: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_992(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:4~0
.
_user_specified_nameweights_intermediate:/}+
)
_user_specified_namefalse_negatives:/|+
)
_user_specified_namefalse_positives:.{*
(
_user_specified_nametrue_positives:6z2
0
_user_specified_nameweights_intermediate_1:1y-
+
_user_specified_namefalse_negatives_1:1x-
+
_user_specified_namefalse_positives_1:0w,
*
_user_specified_nametrue_positives_1:1v-
+
_user_specified_namefalse_positives_2:0u,
*
_user_specified_nametrue_positives_2:1t-
+
_user_specified_namefalse_negatives_2:0s,
*
_user_specified_nametrue_positives_3:1r-
+
_user_specified_namefalse_positives_3:0q,
*
_user_specified_nametrue_positives_4:1p-
+
_user_specified_namefalse_negatives_3:0o,
*
_user_specified_nametrue_positives_5:1n-
+
_user_specified_namefalse_positives_4:0m,
*
_user_specified_nametrue_positives_6:1l-
+
_user_specified_namefalse_negatives_4:0k,
*
_user_specified_nametrue_positives_7:1j-
+
_user_specified_namefalse_positives_5:0i,
*
_user_specified_nametrue_positives_8:1h-
+
_user_specified_namefalse_negatives_5:0g,
*
_user_specified_nametrue_positives_9:1f-
+
_user_specified_namefalse_positives_6:1e-
+
_user_specified_nametrue_positives_10:1d-
+
_user_specified_namefalse_negatives_6:1c-
+
_user_specified_nametrue_positives_11:1b-
+
_user_specified_namefalse_positives_7:1a-
+
_user_specified_nametrue_positives_12:1`-
+
_user_specified_namefalse_negatives_7:1_-
+
_user_specified_nametrue_positives_13:1^-
+
_user_specified_namefalse_positives_8:1]-
+
_user_specified_nametrue_positives_14:1\-
+
_user_specified_namefalse_negatives_8:1[-
+
_user_specified_nametrue_positives_15:1Z-
+
_user_specified_namefalse_positives_9:1Y-
+
_user_specified_nametrue_positives_16:1X-
+
_user_specified_namefalse_negatives_9:1W-
+
_user_specified_nametrue_positives_17:2V.
,
_user_specified_namefalse_positives_10:1U-
+
_user_specified_nametrue_positives_18:2T.
,
_user_specified_namefalse_negatives_10:1S-
+
_user_specified_nametrue_positives_19:2R.
,
_user_specified_namefalse_positives_11:1Q-
+
_user_specified_nametrue_positives_20:2P.
,
_user_specified_namefalse_negatives_11:1O-
+
_user_specified_nametrue_positives_21:2N.
,
_user_specified_namefalse_positives_12:1M-
+
_user_specified_nametrue_positives_22:2L.
,
_user_specified_namefalse_negatives_12:1K-
+
_user_specified_nametrue_positives_23:2J.
,
_user_specified_namefalse_positives_13:1I-
+
_user_specified_nametrue_positives_24:2H.
,
_user_specified_namefalse_negatives_13:1G-
+
_user_specified_nametrue_positives_25:2F.
,
_user_specified_namefalse_positives_14:1E-
+
_user_specified_nametrue_positives_26:2D.
,
_user_specified_namefalse_negatives_14:1C-
+
_user_specified_nametrue_positives_27:2B.
,
_user_specified_namefalse_positives_15:1A-
+
_user_specified_nametrue_positives_28:2@.
,
_user_specified_namefalse_negatives_15:1?-
+
_user_specified_nametrue_positives_29:2>.
,
_user_specified_namefalse_positives_16:1=-
+
_user_specified_nametrue_positives_30:2<.
,
_user_specified_namefalse_negatives_16:1;-
+
_user_specified_nametrue_positives_31:2:.
,
_user_specified_namefalse_negatives_17:29.
,
_user_specified_namefalse_positives_17:.8*
(
_user_specified_nametrue_negatives:17-
+
_user_specified_nametrue_positives_32:%6!

_user_specified_namecount:%5!

_user_specified_nametotal:'4#
!
_user_specified_name	count_1:'3#
!
_user_specified_name	total_1:32/
-
_user_specified_nameAdam/v/dense_4/bias:31/
-
_user_specified_nameAdam/m/dense_4/bias:501
/
_user_specified_nameAdam/v/dense_4/kernel:5/1
/
_user_specified_nameAdam/m/dense_4/kernel:3./
-
_user_specified_nameAdam/v/dense_3/bias:3-/
-
_user_specified_nameAdam/m/dense_3/bias:5,1
/
_user_specified_nameAdam/v/dense_3/kernel:5+1
/
_user_specified_nameAdam/m/dense_3/kernel:3*/
-
_user_specified_nameAdam/v/dense_2/bias:3)/
-
_user_specified_nameAdam/m/dense_2/bias:5(1
/
_user_specified_nameAdam/v/dense_2/kernel:5'1
/
_user_specified_nameAdam/m/dense_2/kernel:3&/
-
_user_specified_nameAdam/v/dense_1/bias:3%/
-
_user_specified_nameAdam/m/dense_1/bias:5$1
/
_user_specified_nameAdam/v/dense_1/kernel:5#1
/
_user_specified_nameAdam/m/dense_1/kernel:1"-
+
_user_specified_nameAdam/v/dense/bias:1!-
+
_user_specified_nameAdam/m/dense/bias:3 /
-
_user_specified_nameAdam/v/dense/kernel:3/
-
_user_specified_nameAdam/m/dense/kernel:=9
7
_user_specified_nameAdam/v/update_connection/bias:=9
7
_user_specified_nameAdam/m/update_connection/bias:IE
C
_user_specified_name+)Adam/v/update_connection/recurrent_kernel:IE
C
_user_specified_name+)Adam/m/update_connection/recurrent_kernel:?;
9
_user_specified_name!Adam/v/update_connection/kernel:?;
9
_user_specified_name!Adam/m/update_connection/kernel:51
/
_user_specified_nameAdam/v/update_ip/bias:51
/
_user_specified_nameAdam/m/update_ip/bias:A=
;
_user_specified_name#!Adam/v/update_ip/recurrent_kernel:A=
;
_user_specified_name#!Adam/m/update_ip/recurrent_kernel:73
1
_user_specified_nameAdam/v/update_ip/kernel:73
1
_user_specified_nameAdam/m/update_ip/kernel:51
/
_user_specified_namecurrent_learning_rate:)%
#
_user_specified_name	iteration:,(
&
_user_specified_namedense_4/bias:.*
(
_user_specified_namedense_4/kernel:,(
&
_user_specified_namedense_3/bias:.*
(
_user_specified_namedense_3/kernel:,(
&
_user_specified_namedense_2/bias:.*
(
_user_specified_namedense_2/kernel:,
(
&
_user_specified_namedense_1/bias:.	*
(
_user_specified_namedense_1/kernel:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:62
0
_user_specified_nameupdate_connection/bias:B>
<
_user_specified_name$"update_connection/recurrent_kernel:84
2
_user_specified_nameupdate_connection/kernel:.*
(
_user_specified_nameupdate_ip/bias::6
4
_user_specified_nameupdate_ip/recurrent_kernel:0,
*
_user_specified_nameupdate_ip/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
б
G
+__inference_dropout_3_layer_call_fn_4847839

inputs
identity▒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_4845679`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Їч	
Р%
__inference_call_557254
inputs_6	
inputs_4	

inputs
inputs_2	
inputs_1	
inputs_5	
inputs_3	C
/sequential_dense_matmul_readvariableop_resource:
АА?
0sequential_dense_biasadd_readvariableop_resource:	АG
3sequential_1_dense_1_matmul_readvariableop_resource:
ААC
4sequential_1_dense_1_biasadd_readvariableop_resource:	А4
!update_ip_readvariableop_resource:	А<
(update_ip_matmul_readvariableop_resource:
АА>
*update_ip_matmul_1_readvariableop_resource:
АА<
)update_connection_readvariableop_resource:	АD
0update_connection_matmul_readvariableop_resource:
ААF
2update_connection_matmul_1_readvariableop_resource:
ААG
3sequential_2_dense_2_matmul_readvariableop_resource:
ААC
4sequential_2_dense_2_biasadd_readvariableop_resource:	АF
3sequential_2_dense_3_matmul_readvariableop_resource:	А@B
4sequential_2_dense_3_biasadd_readvariableop_resource:@E
3sequential_2_dense_4_matmul_readvariableop_resource:@B
4sequential_2_dense_4_biasadd_readvariableop_resource:
identityИв'sequential/dense/BiasAdd/ReadVariableOpв)sequential/dense/BiasAdd_1/ReadVariableOpв)sequential/dense/BiasAdd_2/ReadVariableOpв)sequential/dense/BiasAdd_3/ReadVariableOpв)sequential/dense/BiasAdd_4/ReadVariableOpв)sequential/dense/BiasAdd_5/ReadVariableOpв)sequential/dense/BiasAdd_6/ReadVariableOpв)sequential/dense/BiasAdd_7/ReadVariableOpв&sequential/dense/MatMul/ReadVariableOpв(sequential/dense/MatMul_1/ReadVariableOpв(sequential/dense/MatMul_2/ReadVariableOpв(sequential/dense/MatMul_3/ReadVariableOpв(sequential/dense/MatMul_4/ReadVariableOpв(sequential/dense/MatMul_5/ReadVariableOpв(sequential/dense/MatMul_6/ReadVariableOpв(sequential/dense/MatMul_7/ReadVariableOpв+sequential_1/dense_1/BiasAdd/ReadVariableOpв-sequential_1/dense_1/BiasAdd_1/ReadVariableOpв-sequential_1/dense_1/BiasAdd_2/ReadVariableOpв-sequential_1/dense_1/BiasAdd_3/ReadVariableOpв-sequential_1/dense_1/BiasAdd_4/ReadVariableOpв-sequential_1/dense_1/BiasAdd_5/ReadVariableOpв-sequential_1/dense_1/BiasAdd_6/ReadVariableOpв-sequential_1/dense_1/BiasAdd_7/ReadVariableOpв*sequential_1/dense_1/MatMul/ReadVariableOpв,sequential_1/dense_1/MatMul_1/ReadVariableOpв,sequential_1/dense_1/MatMul_2/ReadVariableOpв,sequential_1/dense_1/MatMul_3/ReadVariableOpв,sequential_1/dense_1/MatMul_4/ReadVariableOpв,sequential_1/dense_1/MatMul_5/ReadVariableOpв,sequential_1/dense_1/MatMul_6/ReadVariableOpв,sequential_1/dense_1/MatMul_7/ReadVariableOpв+sequential_2/dense_2/BiasAdd/ReadVariableOpв*sequential_2/dense_2/MatMul/ReadVariableOpв+sequential_2/dense_3/BiasAdd/ReadVariableOpв*sequential_2/dense_3/MatMul/ReadVariableOpв+sequential_2/dense_4/BiasAdd/ReadVariableOpв*sequential_2/dense_4/MatMul/ReadVariableOpв'update_connection/MatMul/ReadVariableOpв)update_connection/MatMul_1/ReadVariableOpв*update_connection/MatMul_10/ReadVariableOpв*update_connection/MatMul_11/ReadVariableOpв*update_connection/MatMul_12/ReadVariableOpв*update_connection/MatMul_13/ReadVariableOpв*update_connection/MatMul_14/ReadVariableOpв*update_connection/MatMul_15/ReadVariableOpв)update_connection/MatMul_2/ReadVariableOpв)update_connection/MatMul_3/ReadVariableOpв)update_connection/MatMul_4/ReadVariableOpв)update_connection/MatMul_5/ReadVariableOpв)update_connection/MatMul_6/ReadVariableOpв)update_connection/MatMul_7/ReadVariableOpв)update_connection/MatMul_8/ReadVariableOpв)update_connection/MatMul_9/ReadVariableOpв update_connection/ReadVariableOpв"update_connection/ReadVariableOp_1в"update_connection/ReadVariableOp_2в"update_connection/ReadVariableOp_3в"update_connection/ReadVariableOp_4в"update_connection/ReadVariableOp_5в"update_connection/ReadVariableOp_6в"update_connection/ReadVariableOp_7вupdate_ip/MatMul/ReadVariableOpв!update_ip/MatMul_1/ReadVariableOpв"update_ip/MatMul_10/ReadVariableOpв"update_ip/MatMul_11/ReadVariableOpв"update_ip/MatMul_12/ReadVariableOpв"update_ip/MatMul_13/ReadVariableOpв"update_ip/MatMul_14/ReadVariableOpв"update_ip/MatMul_15/ReadVariableOpв!update_ip/MatMul_2/ReadVariableOpв!update_ip/MatMul_3/ReadVariableOpв!update_ip/MatMul_4/ReadVariableOpв!update_ip/MatMul_5/ReadVariableOpв!update_ip/MatMul_6/ReadVariableOpв!update_ip/MatMul_7/ReadVariableOpв!update_ip/MatMul_8/ReadVariableOpв!update_ip/MatMul_9/ReadVariableOpвupdate_ip/ReadVariableOpвupdate_ip/ReadVariableOp_1вupdate_ip/ReadVariableOp_2вupdate_ip/ReadVariableOp_3вupdate_ip/ReadVariableOp_4вupdate_ip/ReadVariableOp_5вupdate_ip/ReadVariableOp_6вupdate_ip/ReadVariableOp_7=
SqueezeSqueezeinputs*
T0*
_output_shapes
:A
	Squeeze_1Squeezeinputs_3*
T0	*
_output_shapes
:A
	Squeeze_2Squeezeinputs_4*
T0	*
_output_shapes
:A
	Squeeze_3Squeezeinputs_5*
T0	*
_output_shapes
:A
	Squeeze_4Squeezeinputs_6*
T0	*
_output_shapes
:K
	ones/CastCastinputs_1*

DstT0*

SrcT0	*
_output_shapes
: P
ones/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Аh
ones/packedPackones/Cast:y:0ones/packed/1:output:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?j
onesFillones/packed:output:0ones/Const:output:0*
T0*(
_output_shapes
:         АI
stack/1Const*
_output_shapes
: *
dtype0	*
value	B	 RfW
stackPackinputs_2stack/1:output:0*
N*
T0	*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    w
zerosFillstack:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         f*

index_type0	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :О
concatConcatV2Squeeze:output:0zeros:output:0concat/axis:output:0*
N*
T0*0
_output_shapes
:                  O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Х
GatherV2GatherV2ones:output:0Squeeze_1:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:Q
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ы

GatherV2_1GatherV2concat:output:0Squeeze_2:output:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:L
	Squeeze_5SqueezeGatherV2_1:output:0*
T0*
_output_shapes
:J
	Squeeze_6SqueezeGatherV2:output:0*
T0*
_output_shapes
:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :А
concat_1ConcatV2Squeeze_6:output:0Squeeze_5:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:
EnsureShapeEnsureShapeconcat_1:output:0*
T0*(
_output_shapes
:         А*
shape:         Аp
sequential/dropout/IdentityIdentityEnsureShape:output:0*
T0*(
_output_shapes
:         АШ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0к
sequential/dense/MatMulMatMul$sequential/dropout/Identity:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АХ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0к
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         Аr
UnsortedSegmentMean/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧c
UnsortedSegmentMean/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Р
UnsortedSegmentMean/onesFill"UnsortedSegmentMean/Shape:output:0'UnsortedSegmentMean/ones/Const:output:0*
T0*
_output_shapes
:─
&UnsortedSegmentMean/UnsortedSegmentSumUnsortedSegmentSum!UnsortedSegmentMean/ones:output:0Squeeze_2:output:0inputs_2*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:q
'UnsortedSegmentMean/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)UnsortedSegmentMean/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: s
)UnsortedSegmentMean/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ь
!UnsortedSegmentMean/strided_sliceStridedSliceinputs_20UnsortedSegmentMean/strided_slice/stack:output:02UnsortedSegmentMean/strided_slice/stack_1:output:02UnsortedSegmentMean/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_maskZ
UnsortedSegmentMean/RankConst*
_output_shapes
: *
dtype0*
value	B :W
UnsortedSegmentMean/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: З
UnsortedSegmentMean/subSub!UnsortedSegmentMean/Rank:output:0#UnsortedSegmentMean/Rank_1:output:0*
T0*
_output_shapes
: t
!UnsortedSegmentMean/ones_1/packedPackUnsortedSegmentMean/sub:z:0*
N*
T0*
_output_shapes
:b
 UnsortedSegmentMean/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rз
UnsortedSegmentMean/ones_1Fill*UnsortedSegmentMean/ones_1/packed:output:0)UnsortedSegmentMean/ones_1/Const:output:0*
T0	*#
_output_shapes
:         a
UnsortedSegmentMean/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╪
UnsortedSegmentMean/concatConcatV2*UnsortedSegmentMean/strided_slice:output:0#UnsortedSegmentMean/ones_1:output:0(UnsortedSegmentMean/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         н
UnsortedSegmentMean/ReshapeReshape/UnsortedSegmentMean/UnsortedSegmentSum:output:0#UnsortedSegmentMean/concat:output:0*
Tshape0	*
T0*
_output_shapes
:b
UnsortedSegmentMean/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ч
UnsortedSegmentMean/MaximumMaximum$UnsortedSegmentMean/Reshape:output:0&UnsortedSegmentMean/Maximum/y:output:0*
T0*
_output_shapes
:╚
(UnsortedSegmentMean/UnsortedSegmentSum_1UnsortedSegmentSum#sequential/dense/Relu:activations:0Squeeze_2:output:0inputs_2*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:Э
UnsortedSegmentMean/truedivRealDiv1UnsortedSegmentMean/UnsortedSegmentSum_1:output:0UnsortedSegmentMean/Maximum:z:0*
T0*
_output_shapes
:Q
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ы

GatherV2_2GatherV2concat:output:0Squeeze_3:output:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:Q
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : Щ

GatherV2_3GatherV2ones:output:0Squeeze_4:output:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:L
	Squeeze_7SqueezeGatherV2_3:output:0*
T0*
_output_shapes
:L
	Squeeze_8SqueezeGatherV2_2:output:0*
T0*
_output_shapes
:O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :А
concat_2ConcatV2Squeeze_8:output:0Squeeze_7:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:Б
EnsureShape_1EnsureShapeconcat_2:output:0*
T0*(
_output_shapes
:         А*
shape:         Аv
sequential_1/dropout_1/IdentityIdentityEnsureShape_1:output:0*
T0*(
_output_shapes
:         Аа
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╢
sequential_1/dense_1/MatMulMatMul(sequential_1/dropout_1/Identity:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЭ
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╢
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А{
sequential_1/dense_1/ReluRelu%sequential_1/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:         Аt
UnsortedSegmentMean_1/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_1/onesFill$UnsortedSegmentMean_1/Shape:output:0)UnsortedSegmentMean_1/ones/Const:output:0*
T0*
_output_shapes
:╚
(UnsortedSegmentMean_1/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_1/ones:output:0Squeeze_4:output:0inputs_1*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:д
#UnsortedSegmentMean_1/strided_sliceStridedSliceinputs_12UnsortedSegmentMean_1/strided_slice/stack:output:04UnsortedSegmentMean_1/strided_slice/stack_1:output:04UnsortedSegmentMean_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_1/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_1/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_1/subSub#UnsortedSegmentMean_1/Rank:output:0%UnsortedSegmentMean_1/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_1/ones_1/packedPackUnsortedSegmentMean_1/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_1/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_1/ones_1Fill,UnsortedSegmentMean_1/ones_1/packed:output:0+UnsortedSegmentMean_1/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_1/concatConcatV2,UnsortedSegmentMean_1/strided_slice:output:0%UnsortedSegmentMean_1/ones_1:output:0*UnsortedSegmentMean_1/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_1/ReshapeReshape1UnsortedSegmentMean_1/UnsortedSegmentSum:output:0%UnsortedSegmentMean_1/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_1/MaximumMaximum&UnsortedSegmentMean_1/Reshape:output:0(UnsortedSegmentMean_1/Maximum/y:output:0*
T0*
_output_shapes
:╬
*UnsortedSegmentMean_1/UnsortedSegmentSum_1UnsortedSegmentSum'sequential_1/dense_1/Relu:activations:0Squeeze_4:output:0inputs_1*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_1/truedivRealDiv3UnsortedSegmentMean_1/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_1/Maximum:z:0*
T0*
_output_shapes
:С
EnsureShape_2EnsureShape!UnsortedSegmentMean_1/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         А{
update_ip/ReadVariableOpReadVariableOp!update_ip_readvariableop_resource*
_output_shapes
:	А*
dtype0u
update_ip/unstackUnpack update_ip/ReadVariableOp:value:0*
T0*"
_output_shapes
:А:А*	
numК
update_ip/MatMul/ReadVariableOpReadVariableOp(update_ip_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0О
update_ip/MatMulMatMulEnsureShape_2:output:0'update_ip/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЗ
update_ip/BiasAddBiasAddupdate_ip/MatMul:product:0update_ip/unstack:output:0*
T0*(
_output_shapes
:         Аd
update_ip/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ─
update_ip/splitSplit"update_ip/split/split_dim:output:0update_ip/BiasAdd:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitО
!update_ip/MatMul_1/ReadVariableOpReadVariableOp*update_ip_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Й
update_ip/MatMul_1MatMulones:output:0)update_ip/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЛ
update_ip/BiasAdd_1BiasAddupdate_ip/MatMul_1:product:0update_ip/unstack:output:1*
T0*(
_output_shapes
:         Аd
update_ip/ConstConst*
_output_shapes
:*
dtype0*!
valueB"А   А       f
update_ip/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ё
update_ip/split_1SplitVupdate_ip/BiasAdd_1:output:0update_ip/Const:output:0$update_ip/split_1/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_split
update_ip/addAddV2update_ip/split:output:0update_ip/split_1:output:0*
T0*(
_output_shapes
:         Аb
update_ip/SigmoidSigmoidupdate_ip/add:z:0*
T0*(
_output_shapes
:         АБ
update_ip/add_1AddV2update_ip/split:output:1update_ip/split_1:output:1*
T0*(
_output_shapes
:         Аf
update_ip/Sigmoid_1Sigmoidupdate_ip/add_1:z:0*
T0*(
_output_shapes
:         А|
update_ip/mulMulupdate_ip/Sigmoid_1:y:0update_ip/split_1:output:2*
T0*(
_output_shapes
:         Аx
update_ip/add_2AddV2update_ip/split:output:2update_ip/mul:z:0*
T0*(
_output_shapes
:         А^
update_ip/TanhTanhupdate_ip/add_2:z:0*
T0*(
_output_shapes
:         Аo
update_ip/mul_1Mulupdate_ip/Sigmoid:y:0ones:output:0*
T0*(
_output_shapes
:         АT
update_ip/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?x
update_ip/subSubupdate_ip/sub/x:output:0update_ip/Sigmoid:y:0*
T0*(
_output_shapes
:         Аp
update_ip/mul_2Mulupdate_ip/sub:z:0update_ip/Tanh:y:0*
T0*(
_output_shapes
:         Аu
update_ip/add_3AddV2update_ip/mul_1:z:0update_ip/mul_2:z:0*
T0*(
_output_shapes
:         АП
EnsureShape_3EnsureShapeUnsortedSegmentMean/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АЛ
 update_connection/ReadVariableOpReadVariableOp)update_connection_readvariableop_resource*
_output_shapes
:	А*
dtype0Е
update_connection/unstackUnpack(update_connection/ReadVariableOp:value:0*
T0*"
_output_shapes
:А:А*	
numЪ
'update_connection/MatMul/ReadVariableOpReadVariableOp0update_connection_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Ю
update_connection/MatMulMatMulEnsureShape_3:output:0/update_connection/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
update_connection/BiasAddBiasAdd"update_connection/MatMul:product:0"update_connection/unstack:output:0*
T0*(
_output_shapes
:         Аl
!update_connection/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ▄
update_connection/splitSplit*update_connection/split/split_dim:output:0"update_connection/BiasAdd:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЮ
)update_connection/MatMul_1/ReadVariableOpReadVariableOp2update_connection_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Ы
update_connection/MatMul_1MatMulconcat:output:01update_connection/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аг
update_connection/BiasAdd_1BiasAdd$update_connection/MatMul_1:product:0"update_connection/unstack:output:1*
T0*(
_output_shapes
:         Аl
update_connection/ConstConst*
_output_shapes
:*
dtype0*!
valueB"А   А       n
#update_connection/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         С
update_connection/split_1SplitV$update_connection/BiasAdd_1:output:0 update_connection/Const:output:0,update_connection/split_1/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЧ
update_connection/addAddV2 update_connection/split:output:0"update_connection/split_1:output:0*
T0*(
_output_shapes
:         Аr
update_connection/SigmoidSigmoidupdate_connection/add:z:0*
T0*(
_output_shapes
:         АЩ
update_connection/add_1AddV2 update_connection/split:output:1"update_connection/split_1:output:1*
T0*(
_output_shapes
:         Аv
update_connection/Sigmoid_1Sigmoidupdate_connection/add_1:z:0*
T0*(
_output_shapes
:         АФ
update_connection/mulMulupdate_connection/Sigmoid_1:y:0"update_connection/split_1:output:2*
T0*(
_output_shapes
:         АР
update_connection/add_2AddV2 update_connection/split:output:2update_connection/mul:z:0*
T0*(
_output_shapes
:         Аn
update_connection/TanhTanhupdate_connection/add_2:z:0*
T0*(
_output_shapes
:         АБ
update_connection/mul_1Mulupdate_connection/Sigmoid:y:0concat:output:0*
T0*(
_output_shapes
:         А\
update_connection/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Р
update_connection/subSub update_connection/sub/x:output:0update_connection/Sigmoid:y:0*
T0*(
_output_shapes
:         АИ
update_connection/mul_2Mulupdate_connection/sub:z:0update_connection/Tanh:y:0*
T0*(
_output_shapes
:         АН
update_connection/add_3AddV2update_connection/mul_1:z:0update_connection/mul_2:z:0*
T0*(
_output_shapes
:         АQ
GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B : Я

GatherV2_4GatherV2update_ip/add_3:z:0Squeeze_1:output:0GatherV2_4/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:Q
GatherV2_5/axisConst*
_output_shapes
: *
dtype0*
value	B : з

GatherV2_5GatherV2update_connection/add_3:z:0Squeeze_2:output:0GatherV2_5/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:L
	Squeeze_9SqueezeGatherV2_5:output:0*
T0*
_output_shapes
:M

Squeeze_10SqueezeGatherV2_4:output:0*
T0*
_output_shapes
:O
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B :Б
concat_3ConcatV2Squeeze_10:output:0Squeeze_9:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:Б
EnsureShape_4EnsureShapeconcat_3:output:0*
T0*(
_output_shapes
:         А*
shape:         Аt
sequential/dropout/Identity_1IdentityEnsureShape_4:output:0*
T0*(
_output_shapes
:         АЪ
(sequential/dense/MatMul_1/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0░
sequential/dense/MatMul_1MatMul&sequential/dropout/Identity_1:output:00sequential/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЧ
)sequential/dense/BiasAdd_1/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0░
sequential/dense/BiasAdd_1BiasAdd#sequential/dense/MatMul_1:product:01sequential/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аw
sequential/dense/Relu_1Relu#sequential/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:         Аt
UnsortedSegmentMean_2/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_2/onesFill$UnsortedSegmentMean_2/Shape:output:0)UnsortedSegmentMean_2/ones/Const:output:0*
T0*
_output_shapes
:╚
(UnsortedSegmentMean_2/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_2/ones:output:0Squeeze_2:output:0inputs_2*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:д
#UnsortedSegmentMean_2/strided_sliceStridedSliceinputs_22UnsortedSegmentMean_2/strided_slice/stack:output:04UnsortedSegmentMean_2/strided_slice/stack_1:output:04UnsortedSegmentMean_2/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_2/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_2/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_2/subSub#UnsortedSegmentMean_2/Rank:output:0%UnsortedSegmentMean_2/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_2/ones_1/packedPackUnsortedSegmentMean_2/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_2/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_2/ones_1Fill,UnsortedSegmentMean_2/ones_1/packed:output:0+UnsortedSegmentMean_2/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_2/concatConcatV2,UnsortedSegmentMean_2/strided_slice:output:0%UnsortedSegmentMean_2/ones_1:output:0*UnsortedSegmentMean_2/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_2/ReshapeReshape1UnsortedSegmentMean_2/UnsortedSegmentSum:output:0%UnsortedSegmentMean_2/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_2/MaximumMaximum&UnsortedSegmentMean_2/Reshape:output:0(UnsortedSegmentMean_2/Maximum/y:output:0*
T0*
_output_shapes
:╠
*UnsortedSegmentMean_2/UnsortedSegmentSum_1UnsortedSegmentSum%sequential/dense/Relu_1:activations:0Squeeze_2:output:0inputs_2*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_2/truedivRealDiv3UnsortedSegmentMean_2/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_2/Maximum:z:0*
T0*
_output_shapes
:Q
GatherV2_6/axisConst*
_output_shapes
: *
dtype0*
value	B : з

GatherV2_6GatherV2update_connection/add_3:z:0Squeeze_3:output:0GatherV2_6/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:Q
GatherV2_7/axisConst*
_output_shapes
: *
dtype0*
value	B : Я

GatherV2_7GatherV2update_ip/add_3:z:0Squeeze_4:output:0GatherV2_7/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:M

Squeeze_11SqueezeGatherV2_7:output:0*
T0*
_output_shapes
:M

Squeeze_12SqueezeGatherV2_6:output:0*
T0*
_output_shapes
:O
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concat_4ConcatV2Squeeze_12:output:0Squeeze_11:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:Б
EnsureShape_5EnsureShapeconcat_4:output:0*
T0*(
_output_shapes
:         А*
shape:         Аx
!sequential_1/dropout_1/Identity_1IdentityEnsureShape_5:output:0*
T0*(
_output_shapes
:         Ав
,sequential_1/dense_1/MatMul_1/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╝
sequential_1/dense_1/MatMul_1MatMul*sequential_1/dropout_1/Identity_1:output:04sequential_1/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
-sequential_1/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╝
sequential_1/dense_1/BiasAdd_1BiasAdd'sequential_1/dense_1/MatMul_1:product:05sequential_1/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А
sequential_1/dense_1/Relu_1Relu'sequential_1/dense_1/BiasAdd_1:output:0*
T0*(
_output_shapes
:         Аt
UnsortedSegmentMean_3/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_3/onesFill$UnsortedSegmentMean_3/Shape:output:0)UnsortedSegmentMean_3/ones/Const:output:0*
T0*
_output_shapes
:╚
(UnsortedSegmentMean_3/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_3/ones:output:0Squeeze_4:output:0inputs_1*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:д
#UnsortedSegmentMean_3/strided_sliceStridedSliceinputs_12UnsortedSegmentMean_3/strided_slice/stack:output:04UnsortedSegmentMean_3/strided_slice/stack_1:output:04UnsortedSegmentMean_3/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_3/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_3/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_3/subSub#UnsortedSegmentMean_3/Rank:output:0%UnsortedSegmentMean_3/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_3/ones_1/packedPackUnsortedSegmentMean_3/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_3/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_3/ones_1Fill,UnsortedSegmentMean_3/ones_1/packed:output:0+UnsortedSegmentMean_3/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_3/concatConcatV2,UnsortedSegmentMean_3/strided_slice:output:0%UnsortedSegmentMean_3/ones_1:output:0*UnsortedSegmentMean_3/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_3/ReshapeReshape1UnsortedSegmentMean_3/UnsortedSegmentSum:output:0%UnsortedSegmentMean_3/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_3/MaximumMaximum&UnsortedSegmentMean_3/Reshape:output:0(UnsortedSegmentMean_3/Maximum/y:output:0*
T0*
_output_shapes
:╨
*UnsortedSegmentMean_3/UnsortedSegmentSum_1UnsortedSegmentSum)sequential_1/dense_1/Relu_1:activations:0Squeeze_4:output:0inputs_1*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_3/truedivRealDiv3UnsortedSegmentMean_3/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_3/Maximum:z:0*
T0*
_output_shapes
:С
EnsureShape_6EnsureShape!UnsortedSegmentMean_3/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         А}
update_ip/ReadVariableOp_1ReadVariableOp!update_ip_readvariableop_resource*
_output_shapes
:	А*
dtype0y
update_ip/unstack_1Unpack"update_ip/ReadVariableOp_1:value:0*
T0*"
_output_shapes
:А:А*	
numМ
!update_ip/MatMul_2/ReadVariableOpReadVariableOp(update_ip_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Т
update_ip/MatMul_2MatMulEnsureShape_6:output:0)update_ip/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АН
update_ip/BiasAdd_2BiasAddupdate_ip/MatMul_2:product:0update_ip/unstack_1:output:0*
T0*(
_output_shapes
:         Аf
update_ip/split_2/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╩
update_ip/split_2Split$update_ip/split_2/split_dim:output:0update_ip/BiasAdd_2:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitО
!update_ip/MatMul_3/ReadVariableOpReadVariableOp*update_ip_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0П
update_ip/MatMul_3MatMulupdate_ip/add_3:z:0)update_ip/MatMul_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АН
update_ip/BiasAdd_3BiasAddupdate_ip/MatMul_3:product:0update_ip/unstack_1:output:1*
T0*(
_output_shapes
:         Аf
update_ip/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"А   А       f
update_ip/split_3/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         є
update_ip/split_3SplitVupdate_ip/BiasAdd_3:output:0update_ip/Const_1:output:0$update_ip/split_3/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitГ
update_ip/add_4AddV2update_ip/split_2:output:0update_ip/split_3:output:0*
T0*(
_output_shapes
:         Аf
update_ip/Sigmoid_2Sigmoidupdate_ip/add_4:z:0*
T0*(
_output_shapes
:         АГ
update_ip/add_5AddV2update_ip/split_2:output:1update_ip/split_3:output:1*
T0*(
_output_shapes
:         Аf
update_ip/Sigmoid_3Sigmoidupdate_ip/add_5:z:0*
T0*(
_output_shapes
:         А~
update_ip/mul_3Mulupdate_ip/Sigmoid_3:y:0update_ip/split_3:output:2*
T0*(
_output_shapes
:         А|
update_ip/add_6AddV2update_ip/split_2:output:2update_ip/mul_3:z:0*
T0*(
_output_shapes
:         А`
update_ip/Tanh_1Tanhupdate_ip/add_6:z:0*
T0*(
_output_shapes
:         Аw
update_ip/mul_4Mulupdate_ip/Sigmoid_2:y:0update_ip/add_3:z:0*
T0*(
_output_shapes
:         АV
update_ip/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?~
update_ip/sub_1Subupdate_ip/sub_1/x:output:0update_ip/Sigmoid_2:y:0*
T0*(
_output_shapes
:         Аt
update_ip/mul_5Mulupdate_ip/sub_1:z:0update_ip/Tanh_1:y:0*
T0*(
_output_shapes
:         Аu
update_ip/add_7AddV2update_ip/mul_4:z:0update_ip/mul_5:z:0*
T0*(
_output_shapes
:         АС
EnsureShape_7EnsureShape!UnsortedSegmentMean_2/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АН
"update_connection/ReadVariableOp_1ReadVariableOp)update_connection_readvariableop_resource*
_output_shapes
:	А*
dtype0Й
update_connection/unstack_1Unpack*update_connection/ReadVariableOp_1:value:0*
T0*"
_output_shapes
:А:А*	
numЬ
)update_connection/MatMul_2/ReadVariableOpReadVariableOp0update_connection_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0в
update_connection/MatMul_2MatMulEnsureShape_7:output:01update_connection/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ае
update_connection/BiasAdd_2BiasAdd$update_connection/MatMul_2:product:0$update_connection/unstack_1:output:0*
T0*(
_output_shapes
:         Аn
#update_connection/split_2/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         т
update_connection/split_2Split,update_connection/split_2/split_dim:output:0$update_connection/BiasAdd_2:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЮ
)update_connection/MatMul_3/ReadVariableOpReadVariableOp2update_connection_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0з
update_connection/MatMul_3MatMulupdate_connection/add_3:z:01update_connection/MatMul_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ае
update_connection/BiasAdd_3BiasAdd$update_connection/MatMul_3:product:0$update_connection/unstack_1:output:1*
T0*(
_output_shapes
:         Аn
update_connection/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"А   А       n
#update_connection/split_3/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         У
update_connection/split_3SplitV$update_connection/BiasAdd_3:output:0"update_connection/Const_1:output:0,update_connection/split_3/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЫ
update_connection/add_4AddV2"update_connection/split_2:output:0"update_connection/split_3:output:0*
T0*(
_output_shapes
:         Аv
update_connection/Sigmoid_2Sigmoidupdate_connection/add_4:z:0*
T0*(
_output_shapes
:         АЫ
update_connection/add_5AddV2"update_connection/split_2:output:1"update_connection/split_3:output:1*
T0*(
_output_shapes
:         Аv
update_connection/Sigmoid_3Sigmoidupdate_connection/add_5:z:0*
T0*(
_output_shapes
:         АЦ
update_connection/mul_3Mulupdate_connection/Sigmoid_3:y:0"update_connection/split_3:output:2*
T0*(
_output_shapes
:         АФ
update_connection/add_6AddV2"update_connection/split_2:output:2update_connection/mul_3:z:0*
T0*(
_output_shapes
:         Аp
update_connection/Tanh_1Tanhupdate_connection/add_6:z:0*
T0*(
_output_shapes
:         АП
update_connection/mul_4Mulupdate_connection/Sigmoid_2:y:0update_connection/add_3:z:0*
T0*(
_output_shapes
:         А^
update_connection/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
update_connection/sub_1Sub"update_connection/sub_1/x:output:0update_connection/Sigmoid_2:y:0*
T0*(
_output_shapes
:         АМ
update_connection/mul_5Mulupdate_connection/sub_1:z:0update_connection/Tanh_1:y:0*
T0*(
_output_shapes
:         АН
update_connection/add_7AddV2update_connection/mul_4:z:0update_connection/mul_5:z:0*
T0*(
_output_shapes
:         АQ
GatherV2_8/axisConst*
_output_shapes
: *
dtype0*
value	B : Я

GatherV2_8GatherV2update_ip/add_7:z:0Squeeze_1:output:0GatherV2_8/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:Q
GatherV2_9/axisConst*
_output_shapes
: *
dtype0*
value	B : з

GatherV2_9GatherV2update_connection/add_7:z:0Squeeze_2:output:0GatherV2_9/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:M

Squeeze_13SqueezeGatherV2_9:output:0*
T0*
_output_shapes
:M

Squeeze_14SqueezeGatherV2_8:output:0*
T0*
_output_shapes
:O
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concat_5ConcatV2Squeeze_14:output:0Squeeze_13:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes
:Б
EnsureShape_8EnsureShapeconcat_5:output:0*
T0*(
_output_shapes
:         А*
shape:         Аt
sequential/dropout/Identity_2IdentityEnsureShape_8:output:0*
T0*(
_output_shapes
:         АЪ
(sequential/dense/MatMul_2/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0░
sequential/dense/MatMul_2MatMul&sequential/dropout/Identity_2:output:00sequential/dense/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЧ
)sequential/dense/BiasAdd_2/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0░
sequential/dense/BiasAdd_2BiasAdd#sequential/dense/MatMul_2:product:01sequential/dense/BiasAdd_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аw
sequential/dense/Relu_2Relu#sequential/dense/BiasAdd_2:output:0*
T0*(
_output_shapes
:         Аt
UnsortedSegmentMean_4/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_4/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_4/onesFill$UnsortedSegmentMean_4/Shape:output:0)UnsortedSegmentMean_4/ones/Const:output:0*
T0*
_output_shapes
:╚
(UnsortedSegmentMean_4/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_4/ones:output:0Squeeze_2:output:0inputs_2*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:д
#UnsortedSegmentMean_4/strided_sliceStridedSliceinputs_22UnsortedSegmentMean_4/strided_slice/stack:output:04UnsortedSegmentMean_4/strided_slice/stack_1:output:04UnsortedSegmentMean_4/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_4/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_4/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_4/subSub#UnsortedSegmentMean_4/Rank:output:0%UnsortedSegmentMean_4/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_4/ones_1/packedPackUnsortedSegmentMean_4/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_4/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_4/ones_1Fill,UnsortedSegmentMean_4/ones_1/packed:output:0+UnsortedSegmentMean_4/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_4/concatConcatV2,UnsortedSegmentMean_4/strided_slice:output:0%UnsortedSegmentMean_4/ones_1:output:0*UnsortedSegmentMean_4/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_4/ReshapeReshape1UnsortedSegmentMean_4/UnsortedSegmentSum:output:0%UnsortedSegmentMean_4/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_4/MaximumMaximum&UnsortedSegmentMean_4/Reshape:output:0(UnsortedSegmentMean_4/Maximum/y:output:0*
T0*
_output_shapes
:╠
*UnsortedSegmentMean_4/UnsortedSegmentSum_1UnsortedSegmentSum%sequential/dense/Relu_2:activations:0Squeeze_2:output:0inputs_2*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_4/truedivRealDiv3UnsortedSegmentMean_4/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_4/Maximum:z:0*
T0*
_output_shapes
:R
GatherV2_10/axisConst*
_output_shapes
: *
dtype0*
value	B : й
GatherV2_10GatherV2update_connection/add_7:z:0Squeeze_3:output:0GatherV2_10/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_11/axisConst*
_output_shapes
: *
dtype0*
value	B : б
GatherV2_11GatherV2update_ip/add_7:z:0Squeeze_4:output:0GatherV2_11/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_15SqueezeGatherV2_11:output:0*
T0*
_output_shapes
:N

Squeeze_16SqueezeGatherV2_10:output:0*
T0*
_output_shapes
:O
concat_6/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concat_6ConcatV2Squeeze_16:output:0Squeeze_15:output:0concat_6/axis:output:0*
N*
T0*
_output_shapes
:Б
EnsureShape_9EnsureShapeconcat_6:output:0*
T0*(
_output_shapes
:         А*
shape:         Аx
!sequential_1/dropout_1/Identity_2IdentityEnsureShape_9:output:0*
T0*(
_output_shapes
:         Ав
,sequential_1/dense_1/MatMul_2/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╝
sequential_1/dense_1/MatMul_2MatMul*sequential_1/dropout_1/Identity_2:output:04sequential_1/dense_1/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
-sequential_1/dense_1/BiasAdd_2/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╝
sequential_1/dense_1/BiasAdd_2BiasAdd'sequential_1/dense_1/MatMul_2:product:05sequential_1/dense_1/BiasAdd_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А
sequential_1/dense_1/Relu_2Relu'sequential_1/dense_1/BiasAdd_2:output:0*
T0*(
_output_shapes
:         Аt
UnsortedSegmentMean_5/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_5/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_5/onesFill$UnsortedSegmentMean_5/Shape:output:0)UnsortedSegmentMean_5/ones/Const:output:0*
T0*
_output_shapes
:╚
(UnsortedSegmentMean_5/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_5/ones:output:0Squeeze_4:output:0inputs_1*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:д
#UnsortedSegmentMean_5/strided_sliceStridedSliceinputs_12UnsortedSegmentMean_5/strided_slice/stack:output:04UnsortedSegmentMean_5/strided_slice/stack_1:output:04UnsortedSegmentMean_5/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_5/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_5/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_5/subSub#UnsortedSegmentMean_5/Rank:output:0%UnsortedSegmentMean_5/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_5/ones_1/packedPackUnsortedSegmentMean_5/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_5/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_5/ones_1Fill,UnsortedSegmentMean_5/ones_1/packed:output:0+UnsortedSegmentMean_5/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_5/concatConcatV2,UnsortedSegmentMean_5/strided_slice:output:0%UnsortedSegmentMean_5/ones_1:output:0*UnsortedSegmentMean_5/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_5/ReshapeReshape1UnsortedSegmentMean_5/UnsortedSegmentSum:output:0%UnsortedSegmentMean_5/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_5/MaximumMaximum&UnsortedSegmentMean_5/Reshape:output:0(UnsortedSegmentMean_5/Maximum/y:output:0*
T0*
_output_shapes
:╨
*UnsortedSegmentMean_5/UnsortedSegmentSum_1UnsortedSegmentSum)sequential_1/dense_1/Relu_2:activations:0Squeeze_4:output:0inputs_1*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_5/truedivRealDiv3UnsortedSegmentMean_5/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_5/Maximum:z:0*
T0*
_output_shapes
:Т
EnsureShape_10EnsureShape!UnsortedSegmentMean_5/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         А}
update_ip/ReadVariableOp_2ReadVariableOp!update_ip_readvariableop_resource*
_output_shapes
:	А*
dtype0y
update_ip/unstack_2Unpack"update_ip/ReadVariableOp_2:value:0*
T0*"
_output_shapes
:А:А*	
numМ
!update_ip/MatMul_4/ReadVariableOpReadVariableOp(update_ip_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0У
update_ip/MatMul_4MatMulEnsureShape_10:output:0)update_ip/MatMul_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АН
update_ip/BiasAdd_4BiasAddupdate_ip/MatMul_4:product:0update_ip/unstack_2:output:0*
T0*(
_output_shapes
:         Аf
update_ip/split_4/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╩
update_ip/split_4Split$update_ip/split_4/split_dim:output:0update_ip/BiasAdd_4:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitО
!update_ip/MatMul_5/ReadVariableOpReadVariableOp*update_ip_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0П
update_ip/MatMul_5MatMulupdate_ip/add_7:z:0)update_ip/MatMul_5/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АН
update_ip/BiasAdd_5BiasAddupdate_ip/MatMul_5:product:0update_ip/unstack_2:output:1*
T0*(
_output_shapes
:         Аf
update_ip/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"А   А       f
update_ip/split_5/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         є
update_ip/split_5SplitVupdate_ip/BiasAdd_5:output:0update_ip/Const_2:output:0$update_ip/split_5/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitГ
update_ip/add_8AddV2update_ip/split_4:output:0update_ip/split_5:output:0*
T0*(
_output_shapes
:         Аf
update_ip/Sigmoid_4Sigmoidupdate_ip/add_8:z:0*
T0*(
_output_shapes
:         АГ
update_ip/add_9AddV2update_ip/split_4:output:1update_ip/split_5:output:1*
T0*(
_output_shapes
:         Аf
update_ip/Sigmoid_5Sigmoidupdate_ip/add_9:z:0*
T0*(
_output_shapes
:         А~
update_ip/mul_6Mulupdate_ip/Sigmoid_5:y:0update_ip/split_5:output:2*
T0*(
_output_shapes
:         А}
update_ip/add_10AddV2update_ip/split_4:output:2update_ip/mul_6:z:0*
T0*(
_output_shapes
:         Аa
update_ip/Tanh_2Tanhupdate_ip/add_10:z:0*
T0*(
_output_shapes
:         Аw
update_ip/mul_7Mulupdate_ip/Sigmoid_4:y:0update_ip/add_7:z:0*
T0*(
_output_shapes
:         АV
update_ip/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?~
update_ip/sub_2Subupdate_ip/sub_2/x:output:0update_ip/Sigmoid_4:y:0*
T0*(
_output_shapes
:         Аt
update_ip/mul_8Mulupdate_ip/sub_2:z:0update_ip/Tanh_2:y:0*
T0*(
_output_shapes
:         Аv
update_ip/add_11AddV2update_ip/mul_7:z:0update_ip/mul_8:z:0*
T0*(
_output_shapes
:         АТ
EnsureShape_11EnsureShape!UnsortedSegmentMean_4/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АН
"update_connection/ReadVariableOp_2ReadVariableOp)update_connection_readvariableop_resource*
_output_shapes
:	А*
dtype0Й
update_connection/unstack_2Unpack*update_connection/ReadVariableOp_2:value:0*
T0*"
_output_shapes
:А:А*	
numЬ
)update_connection/MatMul_4/ReadVariableOpReadVariableOp0update_connection_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0г
update_connection/MatMul_4MatMulEnsureShape_11:output:01update_connection/MatMul_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ае
update_connection/BiasAdd_4BiasAdd$update_connection/MatMul_4:product:0$update_connection/unstack_2:output:0*
T0*(
_output_shapes
:         Аn
#update_connection/split_4/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         т
update_connection/split_4Split,update_connection/split_4/split_dim:output:0$update_connection/BiasAdd_4:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЮ
)update_connection/MatMul_5/ReadVariableOpReadVariableOp2update_connection_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0з
update_connection/MatMul_5MatMulupdate_connection/add_7:z:01update_connection/MatMul_5/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ае
update_connection/BiasAdd_5BiasAdd$update_connection/MatMul_5:product:0$update_connection/unstack_2:output:1*
T0*(
_output_shapes
:         Аn
update_connection/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"А   А       n
#update_connection/split_5/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         У
update_connection/split_5SplitV$update_connection/BiasAdd_5:output:0"update_connection/Const_2:output:0,update_connection/split_5/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЫ
update_connection/add_8AddV2"update_connection/split_4:output:0"update_connection/split_5:output:0*
T0*(
_output_shapes
:         Аv
update_connection/Sigmoid_4Sigmoidupdate_connection/add_8:z:0*
T0*(
_output_shapes
:         АЫ
update_connection/add_9AddV2"update_connection/split_4:output:1"update_connection/split_5:output:1*
T0*(
_output_shapes
:         Аv
update_connection/Sigmoid_5Sigmoidupdate_connection/add_9:z:0*
T0*(
_output_shapes
:         АЦ
update_connection/mul_6Mulupdate_connection/Sigmoid_5:y:0"update_connection/split_5:output:2*
T0*(
_output_shapes
:         АХ
update_connection/add_10AddV2"update_connection/split_4:output:2update_connection/mul_6:z:0*
T0*(
_output_shapes
:         Аq
update_connection/Tanh_2Tanhupdate_connection/add_10:z:0*
T0*(
_output_shapes
:         АП
update_connection/mul_7Mulupdate_connection/Sigmoid_4:y:0update_connection/add_7:z:0*
T0*(
_output_shapes
:         А^
update_connection/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
update_connection/sub_2Sub"update_connection/sub_2/x:output:0update_connection/Sigmoid_4:y:0*
T0*(
_output_shapes
:         АМ
update_connection/mul_8Mulupdate_connection/sub_2:z:0update_connection/Tanh_2:y:0*
T0*(
_output_shapes
:         АО
update_connection/add_11AddV2update_connection/mul_7:z:0update_connection/mul_8:z:0*
T0*(
_output_shapes
:         АR
GatherV2_12/axisConst*
_output_shapes
: *
dtype0*
value	B : в
GatherV2_12GatherV2update_ip/add_11:z:0Squeeze_1:output:0GatherV2_12/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_13/axisConst*
_output_shapes
: *
dtype0*
value	B : к
GatherV2_13GatherV2update_connection/add_11:z:0Squeeze_2:output:0GatherV2_13/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_17SqueezeGatherV2_13:output:0*
T0*
_output_shapes
:N

Squeeze_18SqueezeGatherV2_12:output:0*
T0*
_output_shapes
:O
concat_7/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concat_7ConcatV2Squeeze_18:output:0Squeeze_17:output:0concat_7/axis:output:0*
N*
T0*
_output_shapes
:В
EnsureShape_12EnsureShapeconcat_7:output:0*
T0*(
_output_shapes
:         А*
shape:         Аu
sequential/dropout/Identity_3IdentityEnsureShape_12:output:0*
T0*(
_output_shapes
:         АЪ
(sequential/dense/MatMul_3/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0░
sequential/dense/MatMul_3MatMul&sequential/dropout/Identity_3:output:00sequential/dense/MatMul_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЧ
)sequential/dense/BiasAdd_3/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0░
sequential/dense/BiasAdd_3BiasAdd#sequential/dense/MatMul_3:product:01sequential/dense/BiasAdd_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аw
sequential/dense/Relu_3Relu#sequential/dense/BiasAdd_3:output:0*
T0*(
_output_shapes
:         Аt
UnsortedSegmentMean_6/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_6/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_6/onesFill$UnsortedSegmentMean_6/Shape:output:0)UnsortedSegmentMean_6/ones/Const:output:0*
T0*
_output_shapes
:╚
(UnsortedSegmentMean_6/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_6/ones:output:0Squeeze_2:output:0inputs_2*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:д
#UnsortedSegmentMean_6/strided_sliceStridedSliceinputs_22UnsortedSegmentMean_6/strided_slice/stack:output:04UnsortedSegmentMean_6/strided_slice/stack_1:output:04UnsortedSegmentMean_6/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_6/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_6/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_6/subSub#UnsortedSegmentMean_6/Rank:output:0%UnsortedSegmentMean_6/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_6/ones_1/packedPackUnsortedSegmentMean_6/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_6/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_6/ones_1Fill,UnsortedSegmentMean_6/ones_1/packed:output:0+UnsortedSegmentMean_6/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_6/concatConcatV2,UnsortedSegmentMean_6/strided_slice:output:0%UnsortedSegmentMean_6/ones_1:output:0*UnsortedSegmentMean_6/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_6/ReshapeReshape1UnsortedSegmentMean_6/UnsortedSegmentSum:output:0%UnsortedSegmentMean_6/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_6/MaximumMaximum&UnsortedSegmentMean_6/Reshape:output:0(UnsortedSegmentMean_6/Maximum/y:output:0*
T0*
_output_shapes
:╠
*UnsortedSegmentMean_6/UnsortedSegmentSum_1UnsortedSegmentSum%sequential/dense/Relu_3:activations:0Squeeze_2:output:0inputs_2*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_6/truedivRealDiv3UnsortedSegmentMean_6/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_6/Maximum:z:0*
T0*
_output_shapes
:R
GatherV2_14/axisConst*
_output_shapes
: *
dtype0*
value	B : к
GatherV2_14GatherV2update_connection/add_11:z:0Squeeze_3:output:0GatherV2_14/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_15/axisConst*
_output_shapes
: *
dtype0*
value	B : в
GatherV2_15GatherV2update_ip/add_11:z:0Squeeze_4:output:0GatherV2_15/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_19SqueezeGatherV2_15:output:0*
T0*
_output_shapes
:N

Squeeze_20SqueezeGatherV2_14:output:0*
T0*
_output_shapes
:O
concat_8/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concat_8ConcatV2Squeeze_20:output:0Squeeze_19:output:0concat_8/axis:output:0*
N*
T0*
_output_shapes
:В
EnsureShape_13EnsureShapeconcat_8:output:0*
T0*(
_output_shapes
:         А*
shape:         Аy
!sequential_1/dropout_1/Identity_3IdentityEnsureShape_13:output:0*
T0*(
_output_shapes
:         Ав
,sequential_1/dense_1/MatMul_3/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╝
sequential_1/dense_1/MatMul_3MatMul*sequential_1/dropout_1/Identity_3:output:04sequential_1/dense_1/MatMul_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
-sequential_1/dense_1/BiasAdd_3/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╝
sequential_1/dense_1/BiasAdd_3BiasAdd'sequential_1/dense_1/MatMul_3:product:05sequential_1/dense_1/BiasAdd_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А
sequential_1/dense_1/Relu_3Relu'sequential_1/dense_1/BiasAdd_3:output:0*
T0*(
_output_shapes
:         Аt
UnsortedSegmentMean_7/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_7/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_7/onesFill$UnsortedSegmentMean_7/Shape:output:0)UnsortedSegmentMean_7/ones/Const:output:0*
T0*
_output_shapes
:╚
(UnsortedSegmentMean_7/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_7/ones:output:0Squeeze_4:output:0inputs_1*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:д
#UnsortedSegmentMean_7/strided_sliceStridedSliceinputs_12UnsortedSegmentMean_7/strided_slice/stack:output:04UnsortedSegmentMean_7/strided_slice/stack_1:output:04UnsortedSegmentMean_7/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_7/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_7/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_7/subSub#UnsortedSegmentMean_7/Rank:output:0%UnsortedSegmentMean_7/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_7/ones_1/packedPackUnsortedSegmentMean_7/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_7/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_7/ones_1Fill,UnsortedSegmentMean_7/ones_1/packed:output:0+UnsortedSegmentMean_7/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_7/concatConcatV2,UnsortedSegmentMean_7/strided_slice:output:0%UnsortedSegmentMean_7/ones_1:output:0*UnsortedSegmentMean_7/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_7/ReshapeReshape1UnsortedSegmentMean_7/UnsortedSegmentSum:output:0%UnsortedSegmentMean_7/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_7/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_7/MaximumMaximum&UnsortedSegmentMean_7/Reshape:output:0(UnsortedSegmentMean_7/Maximum/y:output:0*
T0*
_output_shapes
:╨
*UnsortedSegmentMean_7/UnsortedSegmentSum_1UnsortedSegmentSum)sequential_1/dense_1/Relu_3:activations:0Squeeze_4:output:0inputs_1*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_7/truedivRealDiv3UnsortedSegmentMean_7/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_7/Maximum:z:0*
T0*
_output_shapes
:Т
EnsureShape_14EnsureShape!UnsortedSegmentMean_7/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         А}
update_ip/ReadVariableOp_3ReadVariableOp!update_ip_readvariableop_resource*
_output_shapes
:	А*
dtype0y
update_ip/unstack_3Unpack"update_ip/ReadVariableOp_3:value:0*
T0*"
_output_shapes
:А:А*	
numМ
!update_ip/MatMul_6/ReadVariableOpReadVariableOp(update_ip_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0У
update_ip/MatMul_6MatMulEnsureShape_14:output:0)update_ip/MatMul_6/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АН
update_ip/BiasAdd_6BiasAddupdate_ip/MatMul_6:product:0update_ip/unstack_3:output:0*
T0*(
_output_shapes
:         Аf
update_ip/split_6/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╩
update_ip/split_6Split$update_ip/split_6/split_dim:output:0update_ip/BiasAdd_6:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitО
!update_ip/MatMul_7/ReadVariableOpReadVariableOp*update_ip_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Р
update_ip/MatMul_7MatMulupdate_ip/add_11:z:0)update_ip/MatMul_7/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АН
update_ip/BiasAdd_7BiasAddupdate_ip/MatMul_7:product:0update_ip/unstack_3:output:1*
T0*(
_output_shapes
:         Аf
update_ip/Const_3Const*
_output_shapes
:*
dtype0*!
valueB"А   А       f
update_ip/split_7/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         є
update_ip/split_7SplitVupdate_ip/BiasAdd_7:output:0update_ip/Const_3:output:0$update_ip/split_7/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitД
update_ip/add_12AddV2update_ip/split_6:output:0update_ip/split_7:output:0*
T0*(
_output_shapes
:         Аg
update_ip/Sigmoid_6Sigmoidupdate_ip/add_12:z:0*
T0*(
_output_shapes
:         АД
update_ip/add_13AddV2update_ip/split_6:output:1update_ip/split_7:output:1*
T0*(
_output_shapes
:         Аg
update_ip/Sigmoid_7Sigmoidupdate_ip/add_13:z:0*
T0*(
_output_shapes
:         А~
update_ip/mul_9Mulupdate_ip/Sigmoid_7:y:0update_ip/split_7:output:2*
T0*(
_output_shapes
:         А}
update_ip/add_14AddV2update_ip/split_6:output:2update_ip/mul_9:z:0*
T0*(
_output_shapes
:         Аa
update_ip/Tanh_3Tanhupdate_ip/add_14:z:0*
T0*(
_output_shapes
:         Аy
update_ip/mul_10Mulupdate_ip/Sigmoid_6:y:0update_ip/add_11:z:0*
T0*(
_output_shapes
:         АV
update_ip/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?~
update_ip/sub_3Subupdate_ip/sub_3/x:output:0update_ip/Sigmoid_6:y:0*
T0*(
_output_shapes
:         Аu
update_ip/mul_11Mulupdate_ip/sub_3:z:0update_ip/Tanh_3:y:0*
T0*(
_output_shapes
:         Аx
update_ip/add_15AddV2update_ip/mul_10:z:0update_ip/mul_11:z:0*
T0*(
_output_shapes
:         АТ
EnsureShape_15EnsureShape!UnsortedSegmentMean_6/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АН
"update_connection/ReadVariableOp_3ReadVariableOp)update_connection_readvariableop_resource*
_output_shapes
:	А*
dtype0Й
update_connection/unstack_3Unpack*update_connection/ReadVariableOp_3:value:0*
T0*"
_output_shapes
:А:А*	
numЬ
)update_connection/MatMul_6/ReadVariableOpReadVariableOp0update_connection_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0г
update_connection/MatMul_6MatMulEnsureShape_15:output:01update_connection/MatMul_6/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ае
update_connection/BiasAdd_6BiasAdd$update_connection/MatMul_6:product:0$update_connection/unstack_3:output:0*
T0*(
_output_shapes
:         Аn
#update_connection/split_6/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         т
update_connection/split_6Split,update_connection/split_6/split_dim:output:0$update_connection/BiasAdd_6:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЮ
)update_connection/MatMul_7/ReadVariableOpReadVariableOp2update_connection_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0и
update_connection/MatMul_7MatMulupdate_connection/add_11:z:01update_connection/MatMul_7/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ае
update_connection/BiasAdd_7BiasAdd$update_connection/MatMul_7:product:0$update_connection/unstack_3:output:1*
T0*(
_output_shapes
:         Аn
update_connection/Const_3Const*
_output_shapes
:*
dtype0*!
valueB"А   А       n
#update_connection/split_7/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         У
update_connection/split_7SplitV$update_connection/BiasAdd_7:output:0"update_connection/Const_3:output:0,update_connection/split_7/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЬ
update_connection/add_12AddV2"update_connection/split_6:output:0"update_connection/split_7:output:0*
T0*(
_output_shapes
:         Аw
update_connection/Sigmoid_6Sigmoidupdate_connection/add_12:z:0*
T0*(
_output_shapes
:         АЬ
update_connection/add_13AddV2"update_connection/split_6:output:1"update_connection/split_7:output:1*
T0*(
_output_shapes
:         Аw
update_connection/Sigmoid_7Sigmoidupdate_connection/add_13:z:0*
T0*(
_output_shapes
:         АЦ
update_connection/mul_9Mulupdate_connection/Sigmoid_7:y:0"update_connection/split_7:output:2*
T0*(
_output_shapes
:         АХ
update_connection/add_14AddV2"update_connection/split_6:output:2update_connection/mul_9:z:0*
T0*(
_output_shapes
:         Аq
update_connection/Tanh_3Tanhupdate_connection/add_14:z:0*
T0*(
_output_shapes
:         АС
update_connection/mul_10Mulupdate_connection/Sigmoid_6:y:0update_connection/add_11:z:0*
T0*(
_output_shapes
:         А^
update_connection/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
update_connection/sub_3Sub"update_connection/sub_3/x:output:0update_connection/Sigmoid_6:y:0*
T0*(
_output_shapes
:         АН
update_connection/mul_11Mulupdate_connection/sub_3:z:0update_connection/Tanh_3:y:0*
T0*(
_output_shapes
:         АР
update_connection/add_15AddV2update_connection/mul_10:z:0update_connection/mul_11:z:0*
T0*(
_output_shapes
:         АR
GatherV2_16/axisConst*
_output_shapes
: *
dtype0*
value	B : в
GatherV2_16GatherV2update_ip/add_15:z:0Squeeze_1:output:0GatherV2_16/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_17/axisConst*
_output_shapes
: *
dtype0*
value	B : к
GatherV2_17GatherV2update_connection/add_15:z:0Squeeze_2:output:0GatherV2_17/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_21SqueezeGatherV2_17:output:0*
T0*
_output_shapes
:N

Squeeze_22SqueezeGatherV2_16:output:0*
T0*
_output_shapes
:O
concat_9/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concat_9ConcatV2Squeeze_22:output:0Squeeze_21:output:0concat_9/axis:output:0*
N*
T0*
_output_shapes
:В
EnsureShape_16EnsureShapeconcat_9:output:0*
T0*(
_output_shapes
:         А*
shape:         Аu
sequential/dropout/Identity_4IdentityEnsureShape_16:output:0*
T0*(
_output_shapes
:         АЪ
(sequential/dense/MatMul_4/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0░
sequential/dense/MatMul_4MatMul&sequential/dropout/Identity_4:output:00sequential/dense/MatMul_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЧ
)sequential/dense/BiasAdd_4/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0░
sequential/dense/BiasAdd_4BiasAdd#sequential/dense/MatMul_4:product:01sequential/dense/BiasAdd_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аw
sequential/dense/Relu_4Relu#sequential/dense/BiasAdd_4:output:0*
T0*(
_output_shapes
:         Аt
UnsortedSegmentMean_8/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_8/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_8/onesFill$UnsortedSegmentMean_8/Shape:output:0)UnsortedSegmentMean_8/ones/Const:output:0*
T0*
_output_shapes
:╚
(UnsortedSegmentMean_8/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_8/ones:output:0Squeeze_2:output:0inputs_2*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:д
#UnsortedSegmentMean_8/strided_sliceStridedSliceinputs_22UnsortedSegmentMean_8/strided_slice/stack:output:04UnsortedSegmentMean_8/strided_slice/stack_1:output:04UnsortedSegmentMean_8/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_8/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_8/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_8/subSub#UnsortedSegmentMean_8/Rank:output:0%UnsortedSegmentMean_8/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_8/ones_1/packedPackUnsortedSegmentMean_8/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_8/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_8/ones_1Fill,UnsortedSegmentMean_8/ones_1/packed:output:0+UnsortedSegmentMean_8/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_8/concatConcatV2,UnsortedSegmentMean_8/strided_slice:output:0%UnsortedSegmentMean_8/ones_1:output:0*UnsortedSegmentMean_8/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_8/ReshapeReshape1UnsortedSegmentMean_8/UnsortedSegmentSum:output:0%UnsortedSegmentMean_8/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_8/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_8/MaximumMaximum&UnsortedSegmentMean_8/Reshape:output:0(UnsortedSegmentMean_8/Maximum/y:output:0*
T0*
_output_shapes
:╠
*UnsortedSegmentMean_8/UnsortedSegmentSum_1UnsortedSegmentSum%sequential/dense/Relu_4:activations:0Squeeze_2:output:0inputs_2*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_8/truedivRealDiv3UnsortedSegmentMean_8/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_8/Maximum:z:0*
T0*
_output_shapes
:R
GatherV2_18/axisConst*
_output_shapes
: *
dtype0*
value	B : к
GatherV2_18GatherV2update_connection/add_15:z:0Squeeze_3:output:0GatherV2_18/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_19/axisConst*
_output_shapes
: *
dtype0*
value	B : в
GatherV2_19GatherV2update_ip/add_15:z:0Squeeze_4:output:0GatherV2_19/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_23SqueezeGatherV2_19:output:0*
T0*
_output_shapes
:N

Squeeze_24SqueezeGatherV2_18:output:0*
T0*
_output_shapes
:P
concat_10/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_10ConcatV2Squeeze_24:output:0Squeeze_23:output:0concat_10/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_17EnsureShapeconcat_10:output:0*
T0*(
_output_shapes
:         А*
shape:         Аy
!sequential_1/dropout_1/Identity_4IdentityEnsureShape_17:output:0*
T0*(
_output_shapes
:         Ав
,sequential_1/dense_1/MatMul_4/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╝
sequential_1/dense_1/MatMul_4MatMul*sequential_1/dropout_1/Identity_4:output:04sequential_1/dense_1/MatMul_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
-sequential_1/dense_1/BiasAdd_4/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╝
sequential_1/dense_1/BiasAdd_4BiasAdd'sequential_1/dense_1/MatMul_4:product:05sequential_1/dense_1/BiasAdd_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А
sequential_1/dense_1/Relu_4Relu'sequential_1/dense_1/BiasAdd_4:output:0*
T0*(
_output_shapes
:         Аt
UnsortedSegmentMean_9/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_9/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_9/onesFill$UnsortedSegmentMean_9/Shape:output:0)UnsortedSegmentMean_9/ones/Const:output:0*
T0*
_output_shapes
:╚
(UnsortedSegmentMean_9/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_9/ones:output:0Squeeze_4:output:0inputs_1*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:д
#UnsortedSegmentMean_9/strided_sliceStridedSliceinputs_12UnsortedSegmentMean_9/strided_slice/stack:output:04UnsortedSegmentMean_9/strided_slice/stack_1:output:04UnsortedSegmentMean_9/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_9/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_9/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_9/subSub#UnsortedSegmentMean_9/Rank:output:0%UnsortedSegmentMean_9/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_9/ones_1/packedPackUnsortedSegmentMean_9/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_9/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_9/ones_1Fill,UnsortedSegmentMean_9/ones_1/packed:output:0+UnsortedSegmentMean_9/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_9/concatConcatV2,UnsortedSegmentMean_9/strided_slice:output:0%UnsortedSegmentMean_9/ones_1:output:0*UnsortedSegmentMean_9/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_9/ReshapeReshape1UnsortedSegmentMean_9/UnsortedSegmentSum:output:0%UnsortedSegmentMean_9/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_9/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_9/MaximumMaximum&UnsortedSegmentMean_9/Reshape:output:0(UnsortedSegmentMean_9/Maximum/y:output:0*
T0*
_output_shapes
:╨
*UnsortedSegmentMean_9/UnsortedSegmentSum_1UnsortedSegmentSum)sequential_1/dense_1/Relu_4:activations:0Squeeze_4:output:0inputs_1*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_9/truedivRealDiv3UnsortedSegmentMean_9/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_9/Maximum:z:0*
T0*
_output_shapes
:Т
EnsureShape_18EnsureShape!UnsortedSegmentMean_9/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         А}
update_ip/ReadVariableOp_4ReadVariableOp!update_ip_readvariableop_resource*
_output_shapes
:	А*
dtype0y
update_ip/unstack_4Unpack"update_ip/ReadVariableOp_4:value:0*
T0*"
_output_shapes
:А:А*	
numМ
!update_ip/MatMul_8/ReadVariableOpReadVariableOp(update_ip_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0У
update_ip/MatMul_8MatMulEnsureShape_18:output:0)update_ip/MatMul_8/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АН
update_ip/BiasAdd_8BiasAddupdate_ip/MatMul_8:product:0update_ip/unstack_4:output:0*
T0*(
_output_shapes
:         Аf
update_ip/split_8/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╩
update_ip/split_8Split$update_ip/split_8/split_dim:output:0update_ip/BiasAdd_8:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitО
!update_ip/MatMul_9/ReadVariableOpReadVariableOp*update_ip_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Р
update_ip/MatMul_9MatMulupdate_ip/add_15:z:0)update_ip/MatMul_9/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АН
update_ip/BiasAdd_9BiasAddupdate_ip/MatMul_9:product:0update_ip/unstack_4:output:1*
T0*(
_output_shapes
:         Аf
update_ip/Const_4Const*
_output_shapes
:*
dtype0*!
valueB"А   А       f
update_ip/split_9/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         є
update_ip/split_9SplitVupdate_ip/BiasAdd_9:output:0update_ip/Const_4:output:0$update_ip/split_9/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitД
update_ip/add_16AddV2update_ip/split_8:output:0update_ip/split_9:output:0*
T0*(
_output_shapes
:         Аg
update_ip/Sigmoid_8Sigmoidupdate_ip/add_16:z:0*
T0*(
_output_shapes
:         АД
update_ip/add_17AddV2update_ip/split_8:output:1update_ip/split_9:output:1*
T0*(
_output_shapes
:         Аg
update_ip/Sigmoid_9Sigmoidupdate_ip/add_17:z:0*
T0*(
_output_shapes
:         А
update_ip/mul_12Mulupdate_ip/Sigmoid_9:y:0update_ip/split_9:output:2*
T0*(
_output_shapes
:         А~
update_ip/add_18AddV2update_ip/split_8:output:2update_ip/mul_12:z:0*
T0*(
_output_shapes
:         Аa
update_ip/Tanh_4Tanhupdate_ip/add_18:z:0*
T0*(
_output_shapes
:         Аy
update_ip/mul_13Mulupdate_ip/Sigmoid_8:y:0update_ip/add_15:z:0*
T0*(
_output_shapes
:         АV
update_ip/sub_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?~
update_ip/sub_4Subupdate_ip/sub_4/x:output:0update_ip/Sigmoid_8:y:0*
T0*(
_output_shapes
:         Аu
update_ip/mul_14Mulupdate_ip/sub_4:z:0update_ip/Tanh_4:y:0*
T0*(
_output_shapes
:         Аx
update_ip/add_19AddV2update_ip/mul_13:z:0update_ip/mul_14:z:0*
T0*(
_output_shapes
:         АТ
EnsureShape_19EnsureShape!UnsortedSegmentMean_8/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АН
"update_connection/ReadVariableOp_4ReadVariableOp)update_connection_readvariableop_resource*
_output_shapes
:	А*
dtype0Й
update_connection/unstack_4Unpack*update_connection/ReadVariableOp_4:value:0*
T0*"
_output_shapes
:А:А*	
numЬ
)update_connection/MatMul_8/ReadVariableOpReadVariableOp0update_connection_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0г
update_connection/MatMul_8MatMulEnsureShape_19:output:01update_connection/MatMul_8/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ае
update_connection/BiasAdd_8BiasAdd$update_connection/MatMul_8:product:0$update_connection/unstack_4:output:0*
T0*(
_output_shapes
:         Аn
#update_connection/split_8/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         т
update_connection/split_8Split,update_connection/split_8/split_dim:output:0$update_connection/BiasAdd_8:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЮ
)update_connection/MatMul_9/ReadVariableOpReadVariableOp2update_connection_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0и
update_connection/MatMul_9MatMulupdate_connection/add_15:z:01update_connection/MatMul_9/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ае
update_connection/BiasAdd_9BiasAdd$update_connection/MatMul_9:product:0$update_connection/unstack_4:output:1*
T0*(
_output_shapes
:         Аn
update_connection/Const_4Const*
_output_shapes
:*
dtype0*!
valueB"А   А       n
#update_connection/split_9/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         У
update_connection/split_9SplitV$update_connection/BiasAdd_9:output:0"update_connection/Const_4:output:0,update_connection/split_9/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЬ
update_connection/add_16AddV2"update_connection/split_8:output:0"update_connection/split_9:output:0*
T0*(
_output_shapes
:         Аw
update_connection/Sigmoid_8Sigmoidupdate_connection/add_16:z:0*
T0*(
_output_shapes
:         АЬ
update_connection/add_17AddV2"update_connection/split_8:output:1"update_connection/split_9:output:1*
T0*(
_output_shapes
:         Аw
update_connection/Sigmoid_9Sigmoidupdate_connection/add_17:z:0*
T0*(
_output_shapes
:         АЧ
update_connection/mul_12Mulupdate_connection/Sigmoid_9:y:0"update_connection/split_9:output:2*
T0*(
_output_shapes
:         АЦ
update_connection/add_18AddV2"update_connection/split_8:output:2update_connection/mul_12:z:0*
T0*(
_output_shapes
:         Аq
update_connection/Tanh_4Tanhupdate_connection/add_18:z:0*
T0*(
_output_shapes
:         АС
update_connection/mul_13Mulupdate_connection/Sigmoid_8:y:0update_connection/add_15:z:0*
T0*(
_output_shapes
:         А^
update_connection/sub_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
update_connection/sub_4Sub"update_connection/sub_4/x:output:0update_connection/Sigmoid_8:y:0*
T0*(
_output_shapes
:         АН
update_connection/mul_14Mulupdate_connection/sub_4:z:0update_connection/Tanh_4:y:0*
T0*(
_output_shapes
:         АР
update_connection/add_19AddV2update_connection/mul_13:z:0update_connection/mul_14:z:0*
T0*(
_output_shapes
:         АR
GatherV2_20/axisConst*
_output_shapes
: *
dtype0*
value	B : в
GatherV2_20GatherV2update_ip/add_19:z:0Squeeze_1:output:0GatherV2_20/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_21/axisConst*
_output_shapes
: *
dtype0*
value	B : к
GatherV2_21GatherV2update_connection/add_19:z:0Squeeze_2:output:0GatherV2_21/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_25SqueezeGatherV2_21:output:0*
T0*
_output_shapes
:N

Squeeze_26SqueezeGatherV2_20:output:0*
T0*
_output_shapes
:P
concat_11/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_11ConcatV2Squeeze_26:output:0Squeeze_25:output:0concat_11/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_20EnsureShapeconcat_11:output:0*
T0*(
_output_shapes
:         А*
shape:         Аu
sequential/dropout/Identity_5IdentityEnsureShape_20:output:0*
T0*(
_output_shapes
:         АЪ
(sequential/dense/MatMul_5/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0░
sequential/dense/MatMul_5MatMul&sequential/dropout/Identity_5:output:00sequential/dense/MatMul_5/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЧ
)sequential/dense/BiasAdd_5/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0░
sequential/dense/BiasAdd_5BiasAdd#sequential/dense/MatMul_5:product:01sequential/dense/BiasAdd_5/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аw
sequential/dense/Relu_5Relu#sequential/dense/BiasAdd_5:output:0*
T0*(
_output_shapes
:         Аu
UnsortedSegmentMean_10/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧f
!UnsortedSegmentMean_10/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
UnsortedSegmentMean_10/onesFill%UnsortedSegmentMean_10/Shape:output:0*UnsortedSegmentMean_10/ones/Const:output:0*
T0*
_output_shapes
:╩
)UnsortedSegmentMean_10/UnsortedSegmentSumUnsortedSegmentSum$UnsortedSegmentMean_10/ones:output:0Squeeze_2:output:0inputs_2*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:t
*UnsortedSegmentMean_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:и
$UnsortedSegmentMean_10/strided_sliceStridedSliceinputs_23UnsortedSegmentMean_10/strided_slice/stack:output:05UnsortedSegmentMean_10/strided_slice/stack_1:output:05UnsortedSegmentMean_10/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask]
UnsortedSegmentMean_10/RankConst*
_output_shapes
: *
dtype0*
value	B :Z
UnsortedSegmentMean_10/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Р
UnsortedSegmentMean_10/subSub$UnsortedSegmentMean_10/Rank:output:0&UnsortedSegmentMean_10/Rank_1:output:0*
T0*
_output_shapes
: z
$UnsortedSegmentMean_10/ones_1/packedPackUnsortedSegmentMean_10/sub:z:0*
N*
T0*
_output_shapes
:e
#UnsortedSegmentMean_10/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R░
UnsortedSegmentMean_10/ones_1Fill-UnsortedSegmentMean_10/ones_1/packed:output:0,UnsortedSegmentMean_10/ones_1/Const:output:0*
T0	*#
_output_shapes
:         d
"UnsortedSegmentMean_10/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
UnsortedSegmentMean_10/concatConcatV2-UnsortedSegmentMean_10/strided_slice:output:0&UnsortedSegmentMean_10/ones_1:output:0+UnsortedSegmentMean_10/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         ╢
UnsortedSegmentMean_10/ReshapeReshape2UnsortedSegmentMean_10/UnsortedSegmentSum:output:0&UnsortedSegmentMean_10/concat:output:0*
Tshape0	*
T0*
_output_shapes
:e
 UnsortedSegmentMean_10/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
UnsortedSegmentMean_10/MaximumMaximum'UnsortedSegmentMean_10/Reshape:output:0)UnsortedSegmentMean_10/Maximum/y:output:0*
T0*
_output_shapes
:═
+UnsortedSegmentMean_10/UnsortedSegmentSum_1UnsortedSegmentSum%sequential/dense/Relu_5:activations:0Squeeze_2:output:0inputs_2*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:ж
UnsortedSegmentMean_10/truedivRealDiv4UnsortedSegmentMean_10/UnsortedSegmentSum_1:output:0"UnsortedSegmentMean_10/Maximum:z:0*
T0*
_output_shapes
:R
GatherV2_22/axisConst*
_output_shapes
: *
dtype0*
value	B : к
GatherV2_22GatherV2update_connection/add_19:z:0Squeeze_3:output:0GatherV2_22/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_23/axisConst*
_output_shapes
: *
dtype0*
value	B : в
GatherV2_23GatherV2update_ip/add_19:z:0Squeeze_4:output:0GatherV2_23/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_27SqueezeGatherV2_23:output:0*
T0*
_output_shapes
:N

Squeeze_28SqueezeGatherV2_22:output:0*
T0*
_output_shapes
:P
concat_12/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_12ConcatV2Squeeze_28:output:0Squeeze_27:output:0concat_12/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_21EnsureShapeconcat_12:output:0*
T0*(
_output_shapes
:         А*
shape:         Аy
!sequential_1/dropout_1/Identity_5IdentityEnsureShape_21:output:0*
T0*(
_output_shapes
:         Ав
,sequential_1/dense_1/MatMul_5/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╝
sequential_1/dense_1/MatMul_5MatMul*sequential_1/dropout_1/Identity_5:output:04sequential_1/dense_1/MatMul_5/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
-sequential_1/dense_1/BiasAdd_5/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╝
sequential_1/dense_1/BiasAdd_5BiasAdd'sequential_1/dense_1/MatMul_5:product:05sequential_1/dense_1/BiasAdd_5/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А
sequential_1/dense_1/Relu_5Relu'sequential_1/dense_1/BiasAdd_5:output:0*
T0*(
_output_shapes
:         Аu
UnsortedSegmentMean_11/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧f
!UnsortedSegmentMean_11/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
UnsortedSegmentMean_11/onesFill%UnsortedSegmentMean_11/Shape:output:0*UnsortedSegmentMean_11/ones/Const:output:0*
T0*
_output_shapes
:╩
)UnsortedSegmentMean_11/UnsortedSegmentSumUnsortedSegmentSum$UnsortedSegmentMean_11/ones:output:0Squeeze_4:output:0inputs_1*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:t
*UnsortedSegmentMean_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:и
$UnsortedSegmentMean_11/strided_sliceStridedSliceinputs_13UnsortedSegmentMean_11/strided_slice/stack:output:05UnsortedSegmentMean_11/strided_slice/stack_1:output:05UnsortedSegmentMean_11/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask]
UnsortedSegmentMean_11/RankConst*
_output_shapes
: *
dtype0*
value	B :Z
UnsortedSegmentMean_11/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Р
UnsortedSegmentMean_11/subSub$UnsortedSegmentMean_11/Rank:output:0&UnsortedSegmentMean_11/Rank_1:output:0*
T0*
_output_shapes
: z
$UnsortedSegmentMean_11/ones_1/packedPackUnsortedSegmentMean_11/sub:z:0*
N*
T0*
_output_shapes
:e
#UnsortedSegmentMean_11/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R░
UnsortedSegmentMean_11/ones_1Fill-UnsortedSegmentMean_11/ones_1/packed:output:0,UnsortedSegmentMean_11/ones_1/Const:output:0*
T0	*#
_output_shapes
:         d
"UnsortedSegmentMean_11/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
UnsortedSegmentMean_11/concatConcatV2-UnsortedSegmentMean_11/strided_slice:output:0&UnsortedSegmentMean_11/ones_1:output:0+UnsortedSegmentMean_11/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         ╢
UnsortedSegmentMean_11/ReshapeReshape2UnsortedSegmentMean_11/UnsortedSegmentSum:output:0&UnsortedSegmentMean_11/concat:output:0*
Tshape0	*
T0*
_output_shapes
:e
 UnsortedSegmentMean_11/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
UnsortedSegmentMean_11/MaximumMaximum'UnsortedSegmentMean_11/Reshape:output:0)UnsortedSegmentMean_11/Maximum/y:output:0*
T0*
_output_shapes
:╤
+UnsortedSegmentMean_11/UnsortedSegmentSum_1UnsortedSegmentSum)sequential_1/dense_1/Relu_5:activations:0Squeeze_4:output:0inputs_1*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:ж
UnsortedSegmentMean_11/truedivRealDiv4UnsortedSegmentMean_11/UnsortedSegmentSum_1:output:0"UnsortedSegmentMean_11/Maximum:z:0*
T0*
_output_shapes
:У
EnsureShape_22EnsureShape"UnsortedSegmentMean_11/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         А}
update_ip/ReadVariableOp_5ReadVariableOp!update_ip_readvariableop_resource*
_output_shapes
:	А*
dtype0y
update_ip/unstack_5Unpack"update_ip/ReadVariableOp_5:value:0*
T0*"
_output_shapes
:А:А*	
numН
"update_ip/MatMul_10/ReadVariableOpReadVariableOp(update_ip_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Х
update_ip/MatMul_10MatMulEnsureShape_22:output:0*update_ip/MatMul_10/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АП
update_ip/BiasAdd_10BiasAddupdate_ip/MatMul_10:product:0update_ip/unstack_5:output:0*
T0*(
_output_shapes
:         Аg
update_ip/split_10/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ═
update_ip/split_10Split%update_ip/split_10/split_dim:output:0update_ip/BiasAdd_10:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitП
"update_ip/MatMul_11/ReadVariableOpReadVariableOp*update_ip_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Т
update_ip/MatMul_11MatMulupdate_ip/add_19:z:0*update_ip/MatMul_11/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АП
update_ip/BiasAdd_11BiasAddupdate_ip/MatMul_11:product:0update_ip/unstack_5:output:1*
T0*(
_output_shapes
:         Аf
update_ip/Const_5Const*
_output_shapes
:*
dtype0*!
valueB"А   А       g
update_ip/split_11/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Ў
update_ip/split_11SplitVupdate_ip/BiasAdd_11:output:0update_ip/Const_5:output:0%update_ip/split_11/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЖ
update_ip/add_20AddV2update_ip/split_10:output:0update_ip/split_11:output:0*
T0*(
_output_shapes
:         Аh
update_ip/Sigmoid_10Sigmoidupdate_ip/add_20:z:0*
T0*(
_output_shapes
:         АЖ
update_ip/add_21AddV2update_ip/split_10:output:1update_ip/split_11:output:1*
T0*(
_output_shapes
:         Аh
update_ip/Sigmoid_11Sigmoidupdate_ip/add_21:z:0*
T0*(
_output_shapes
:         АБ
update_ip/mul_15Mulupdate_ip/Sigmoid_11:y:0update_ip/split_11:output:2*
T0*(
_output_shapes
:         А
update_ip/add_22AddV2update_ip/split_10:output:2update_ip/mul_15:z:0*
T0*(
_output_shapes
:         Аa
update_ip/Tanh_5Tanhupdate_ip/add_22:z:0*
T0*(
_output_shapes
:         Аz
update_ip/mul_16Mulupdate_ip/Sigmoid_10:y:0update_ip/add_19:z:0*
T0*(
_output_shapes
:         АV
update_ip/sub_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
update_ip/sub_5Subupdate_ip/sub_5/x:output:0update_ip/Sigmoid_10:y:0*
T0*(
_output_shapes
:         Аu
update_ip/mul_17Mulupdate_ip/sub_5:z:0update_ip/Tanh_5:y:0*
T0*(
_output_shapes
:         Аx
update_ip/add_23AddV2update_ip/mul_16:z:0update_ip/mul_17:z:0*
T0*(
_output_shapes
:         АУ
EnsureShape_23EnsureShape"UnsortedSegmentMean_10/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АН
"update_connection/ReadVariableOp_5ReadVariableOp)update_connection_readvariableop_resource*
_output_shapes
:	А*
dtype0Й
update_connection/unstack_5Unpack*update_connection/ReadVariableOp_5:value:0*
T0*"
_output_shapes
:А:А*	
numЭ
*update_connection/MatMul_10/ReadVariableOpReadVariableOp0update_connection_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0е
update_connection/MatMul_10MatMulEnsureShape_23:output:02update_connection/MatMul_10/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аз
update_connection/BiasAdd_10BiasAdd%update_connection/MatMul_10:product:0$update_connection/unstack_5:output:0*
T0*(
_output_shapes
:         Аo
$update_connection/split_10/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         х
update_connection/split_10Split-update_connection/split_10/split_dim:output:0%update_connection/BiasAdd_10:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЯ
*update_connection/MatMul_11/ReadVariableOpReadVariableOp2update_connection_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0к
update_connection/MatMul_11MatMulupdate_connection/add_19:z:02update_connection/MatMul_11/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аз
update_connection/BiasAdd_11BiasAdd%update_connection/MatMul_11:product:0$update_connection/unstack_5:output:1*
T0*(
_output_shapes
:         Аn
update_connection/Const_5Const*
_output_shapes
:*
dtype0*!
valueB"А   А       o
$update_connection/split_11/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Ц
update_connection/split_11SplitV%update_connection/BiasAdd_11:output:0"update_connection/Const_5:output:0-update_connection/split_11/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЮ
update_connection/add_20AddV2#update_connection/split_10:output:0#update_connection/split_11:output:0*
T0*(
_output_shapes
:         Аx
update_connection/Sigmoid_10Sigmoidupdate_connection/add_20:z:0*
T0*(
_output_shapes
:         АЮ
update_connection/add_21AddV2#update_connection/split_10:output:1#update_connection/split_11:output:1*
T0*(
_output_shapes
:         Аx
update_connection/Sigmoid_11Sigmoidupdate_connection/add_21:z:0*
T0*(
_output_shapes
:         АЩ
update_connection/mul_15Mul update_connection/Sigmoid_11:y:0#update_connection/split_11:output:2*
T0*(
_output_shapes
:         АЧ
update_connection/add_22AddV2#update_connection/split_10:output:2update_connection/mul_15:z:0*
T0*(
_output_shapes
:         Аq
update_connection/Tanh_5Tanhupdate_connection/add_22:z:0*
T0*(
_output_shapes
:         АТ
update_connection/mul_16Mul update_connection/Sigmoid_10:y:0update_connection/add_19:z:0*
T0*(
_output_shapes
:         А^
update_connection/sub_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ч
update_connection/sub_5Sub"update_connection/sub_5/x:output:0 update_connection/Sigmoid_10:y:0*
T0*(
_output_shapes
:         АН
update_connection/mul_17Mulupdate_connection/sub_5:z:0update_connection/Tanh_5:y:0*
T0*(
_output_shapes
:         АР
update_connection/add_23AddV2update_connection/mul_16:z:0update_connection/mul_17:z:0*
T0*(
_output_shapes
:         АR
GatherV2_24/axisConst*
_output_shapes
: *
dtype0*
value	B : в
GatherV2_24GatherV2update_ip/add_23:z:0Squeeze_1:output:0GatherV2_24/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_25/axisConst*
_output_shapes
: *
dtype0*
value	B : к
GatherV2_25GatherV2update_connection/add_23:z:0Squeeze_2:output:0GatherV2_25/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_29SqueezeGatherV2_25:output:0*
T0*
_output_shapes
:N

Squeeze_30SqueezeGatherV2_24:output:0*
T0*
_output_shapes
:P
concat_13/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_13ConcatV2Squeeze_30:output:0Squeeze_29:output:0concat_13/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_24EnsureShapeconcat_13:output:0*
T0*(
_output_shapes
:         А*
shape:         Аu
sequential/dropout/Identity_6IdentityEnsureShape_24:output:0*
T0*(
_output_shapes
:         АЪ
(sequential/dense/MatMul_6/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0░
sequential/dense/MatMul_6MatMul&sequential/dropout/Identity_6:output:00sequential/dense/MatMul_6/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЧ
)sequential/dense/BiasAdd_6/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0░
sequential/dense/BiasAdd_6BiasAdd#sequential/dense/MatMul_6:product:01sequential/dense/BiasAdd_6/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аw
sequential/dense/Relu_6Relu#sequential/dense/BiasAdd_6:output:0*
T0*(
_output_shapes
:         Аu
UnsortedSegmentMean_12/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧f
!UnsortedSegmentMean_12/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
UnsortedSegmentMean_12/onesFill%UnsortedSegmentMean_12/Shape:output:0*UnsortedSegmentMean_12/ones/Const:output:0*
T0*
_output_shapes
:╩
)UnsortedSegmentMean_12/UnsortedSegmentSumUnsortedSegmentSum$UnsortedSegmentMean_12/ones:output:0Squeeze_2:output:0inputs_2*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:t
*UnsortedSegmentMean_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:и
$UnsortedSegmentMean_12/strided_sliceStridedSliceinputs_23UnsortedSegmentMean_12/strided_slice/stack:output:05UnsortedSegmentMean_12/strided_slice/stack_1:output:05UnsortedSegmentMean_12/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask]
UnsortedSegmentMean_12/RankConst*
_output_shapes
: *
dtype0*
value	B :Z
UnsortedSegmentMean_12/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Р
UnsortedSegmentMean_12/subSub$UnsortedSegmentMean_12/Rank:output:0&UnsortedSegmentMean_12/Rank_1:output:0*
T0*
_output_shapes
: z
$UnsortedSegmentMean_12/ones_1/packedPackUnsortedSegmentMean_12/sub:z:0*
N*
T0*
_output_shapes
:e
#UnsortedSegmentMean_12/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R░
UnsortedSegmentMean_12/ones_1Fill-UnsortedSegmentMean_12/ones_1/packed:output:0,UnsortedSegmentMean_12/ones_1/Const:output:0*
T0	*#
_output_shapes
:         d
"UnsortedSegmentMean_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
UnsortedSegmentMean_12/concatConcatV2-UnsortedSegmentMean_12/strided_slice:output:0&UnsortedSegmentMean_12/ones_1:output:0+UnsortedSegmentMean_12/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         ╢
UnsortedSegmentMean_12/ReshapeReshape2UnsortedSegmentMean_12/UnsortedSegmentSum:output:0&UnsortedSegmentMean_12/concat:output:0*
Tshape0	*
T0*
_output_shapes
:e
 UnsortedSegmentMean_12/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
UnsortedSegmentMean_12/MaximumMaximum'UnsortedSegmentMean_12/Reshape:output:0)UnsortedSegmentMean_12/Maximum/y:output:0*
T0*
_output_shapes
:═
+UnsortedSegmentMean_12/UnsortedSegmentSum_1UnsortedSegmentSum%sequential/dense/Relu_6:activations:0Squeeze_2:output:0inputs_2*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:ж
UnsortedSegmentMean_12/truedivRealDiv4UnsortedSegmentMean_12/UnsortedSegmentSum_1:output:0"UnsortedSegmentMean_12/Maximum:z:0*
T0*
_output_shapes
:R
GatherV2_26/axisConst*
_output_shapes
: *
dtype0*
value	B : к
GatherV2_26GatherV2update_connection/add_23:z:0Squeeze_3:output:0GatherV2_26/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_27/axisConst*
_output_shapes
: *
dtype0*
value	B : в
GatherV2_27GatherV2update_ip/add_23:z:0Squeeze_4:output:0GatherV2_27/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_31SqueezeGatherV2_27:output:0*
T0*
_output_shapes
:N

Squeeze_32SqueezeGatherV2_26:output:0*
T0*
_output_shapes
:P
concat_14/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_14ConcatV2Squeeze_32:output:0Squeeze_31:output:0concat_14/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_25EnsureShapeconcat_14:output:0*
T0*(
_output_shapes
:         А*
shape:         Аy
!sequential_1/dropout_1/Identity_6IdentityEnsureShape_25:output:0*
T0*(
_output_shapes
:         Ав
,sequential_1/dense_1/MatMul_6/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╝
sequential_1/dense_1/MatMul_6MatMul*sequential_1/dropout_1/Identity_6:output:04sequential_1/dense_1/MatMul_6/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
-sequential_1/dense_1/BiasAdd_6/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╝
sequential_1/dense_1/BiasAdd_6BiasAdd'sequential_1/dense_1/MatMul_6:product:05sequential_1/dense_1/BiasAdd_6/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А
sequential_1/dense_1/Relu_6Relu'sequential_1/dense_1/BiasAdd_6:output:0*
T0*(
_output_shapes
:         Аu
UnsortedSegmentMean_13/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧f
!UnsortedSegmentMean_13/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
UnsortedSegmentMean_13/onesFill%UnsortedSegmentMean_13/Shape:output:0*UnsortedSegmentMean_13/ones/Const:output:0*
T0*
_output_shapes
:╩
)UnsortedSegmentMean_13/UnsortedSegmentSumUnsortedSegmentSum$UnsortedSegmentMean_13/ones:output:0Squeeze_4:output:0inputs_1*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:t
*UnsortedSegmentMean_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:и
$UnsortedSegmentMean_13/strided_sliceStridedSliceinputs_13UnsortedSegmentMean_13/strided_slice/stack:output:05UnsortedSegmentMean_13/strided_slice/stack_1:output:05UnsortedSegmentMean_13/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask]
UnsortedSegmentMean_13/RankConst*
_output_shapes
: *
dtype0*
value	B :Z
UnsortedSegmentMean_13/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Р
UnsortedSegmentMean_13/subSub$UnsortedSegmentMean_13/Rank:output:0&UnsortedSegmentMean_13/Rank_1:output:0*
T0*
_output_shapes
: z
$UnsortedSegmentMean_13/ones_1/packedPackUnsortedSegmentMean_13/sub:z:0*
N*
T0*
_output_shapes
:e
#UnsortedSegmentMean_13/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R░
UnsortedSegmentMean_13/ones_1Fill-UnsortedSegmentMean_13/ones_1/packed:output:0,UnsortedSegmentMean_13/ones_1/Const:output:0*
T0	*#
_output_shapes
:         d
"UnsortedSegmentMean_13/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
UnsortedSegmentMean_13/concatConcatV2-UnsortedSegmentMean_13/strided_slice:output:0&UnsortedSegmentMean_13/ones_1:output:0+UnsortedSegmentMean_13/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         ╢
UnsortedSegmentMean_13/ReshapeReshape2UnsortedSegmentMean_13/UnsortedSegmentSum:output:0&UnsortedSegmentMean_13/concat:output:0*
Tshape0	*
T0*
_output_shapes
:e
 UnsortedSegmentMean_13/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
UnsortedSegmentMean_13/MaximumMaximum'UnsortedSegmentMean_13/Reshape:output:0)UnsortedSegmentMean_13/Maximum/y:output:0*
T0*
_output_shapes
:╤
+UnsortedSegmentMean_13/UnsortedSegmentSum_1UnsortedSegmentSum)sequential_1/dense_1/Relu_6:activations:0Squeeze_4:output:0inputs_1*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:ж
UnsortedSegmentMean_13/truedivRealDiv4UnsortedSegmentMean_13/UnsortedSegmentSum_1:output:0"UnsortedSegmentMean_13/Maximum:z:0*
T0*
_output_shapes
:У
EnsureShape_26EnsureShape"UnsortedSegmentMean_13/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         А}
update_ip/ReadVariableOp_6ReadVariableOp!update_ip_readvariableop_resource*
_output_shapes
:	А*
dtype0y
update_ip/unstack_6Unpack"update_ip/ReadVariableOp_6:value:0*
T0*"
_output_shapes
:А:А*	
numН
"update_ip/MatMul_12/ReadVariableOpReadVariableOp(update_ip_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Х
update_ip/MatMul_12MatMulEnsureShape_26:output:0*update_ip/MatMul_12/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АП
update_ip/BiasAdd_12BiasAddupdate_ip/MatMul_12:product:0update_ip/unstack_6:output:0*
T0*(
_output_shapes
:         Аg
update_ip/split_12/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ═
update_ip/split_12Split%update_ip/split_12/split_dim:output:0update_ip/BiasAdd_12:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitП
"update_ip/MatMul_13/ReadVariableOpReadVariableOp*update_ip_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Т
update_ip/MatMul_13MatMulupdate_ip/add_23:z:0*update_ip/MatMul_13/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АП
update_ip/BiasAdd_13BiasAddupdate_ip/MatMul_13:product:0update_ip/unstack_6:output:1*
T0*(
_output_shapes
:         Аf
update_ip/Const_6Const*
_output_shapes
:*
dtype0*!
valueB"А   А       g
update_ip/split_13/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Ў
update_ip/split_13SplitVupdate_ip/BiasAdd_13:output:0update_ip/Const_6:output:0%update_ip/split_13/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЖ
update_ip/add_24AddV2update_ip/split_12:output:0update_ip/split_13:output:0*
T0*(
_output_shapes
:         Аh
update_ip/Sigmoid_12Sigmoidupdate_ip/add_24:z:0*
T0*(
_output_shapes
:         АЖ
update_ip/add_25AddV2update_ip/split_12:output:1update_ip/split_13:output:1*
T0*(
_output_shapes
:         Аh
update_ip/Sigmoid_13Sigmoidupdate_ip/add_25:z:0*
T0*(
_output_shapes
:         АБ
update_ip/mul_18Mulupdate_ip/Sigmoid_13:y:0update_ip/split_13:output:2*
T0*(
_output_shapes
:         А
update_ip/add_26AddV2update_ip/split_12:output:2update_ip/mul_18:z:0*
T0*(
_output_shapes
:         Аa
update_ip/Tanh_6Tanhupdate_ip/add_26:z:0*
T0*(
_output_shapes
:         Аz
update_ip/mul_19Mulupdate_ip/Sigmoid_12:y:0update_ip/add_23:z:0*
T0*(
_output_shapes
:         АV
update_ip/sub_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
update_ip/sub_6Subupdate_ip/sub_6/x:output:0update_ip/Sigmoid_12:y:0*
T0*(
_output_shapes
:         Аu
update_ip/mul_20Mulupdate_ip/sub_6:z:0update_ip/Tanh_6:y:0*
T0*(
_output_shapes
:         Аx
update_ip/add_27AddV2update_ip/mul_19:z:0update_ip/mul_20:z:0*
T0*(
_output_shapes
:         АУ
EnsureShape_27EnsureShape"UnsortedSegmentMean_12/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АН
"update_connection/ReadVariableOp_6ReadVariableOp)update_connection_readvariableop_resource*
_output_shapes
:	А*
dtype0Й
update_connection/unstack_6Unpack*update_connection/ReadVariableOp_6:value:0*
T0*"
_output_shapes
:А:А*	
numЭ
*update_connection/MatMul_12/ReadVariableOpReadVariableOp0update_connection_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0е
update_connection/MatMul_12MatMulEnsureShape_27:output:02update_connection/MatMul_12/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аз
update_connection/BiasAdd_12BiasAdd%update_connection/MatMul_12:product:0$update_connection/unstack_6:output:0*
T0*(
_output_shapes
:         Аo
$update_connection/split_12/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         х
update_connection/split_12Split-update_connection/split_12/split_dim:output:0%update_connection/BiasAdd_12:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЯ
*update_connection/MatMul_13/ReadVariableOpReadVariableOp2update_connection_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0к
update_connection/MatMul_13MatMulupdate_connection/add_23:z:02update_connection/MatMul_13/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аз
update_connection/BiasAdd_13BiasAdd%update_connection/MatMul_13:product:0$update_connection/unstack_6:output:1*
T0*(
_output_shapes
:         Аn
update_connection/Const_6Const*
_output_shapes
:*
dtype0*!
valueB"А   А       o
$update_connection/split_13/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Ц
update_connection/split_13SplitV%update_connection/BiasAdd_13:output:0"update_connection/Const_6:output:0-update_connection/split_13/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЮ
update_connection/add_24AddV2#update_connection/split_12:output:0#update_connection/split_13:output:0*
T0*(
_output_shapes
:         Аx
update_connection/Sigmoid_12Sigmoidupdate_connection/add_24:z:0*
T0*(
_output_shapes
:         АЮ
update_connection/add_25AddV2#update_connection/split_12:output:1#update_connection/split_13:output:1*
T0*(
_output_shapes
:         Аx
update_connection/Sigmoid_13Sigmoidupdate_connection/add_25:z:0*
T0*(
_output_shapes
:         АЩ
update_connection/mul_18Mul update_connection/Sigmoid_13:y:0#update_connection/split_13:output:2*
T0*(
_output_shapes
:         АЧ
update_connection/add_26AddV2#update_connection/split_12:output:2update_connection/mul_18:z:0*
T0*(
_output_shapes
:         Аq
update_connection/Tanh_6Tanhupdate_connection/add_26:z:0*
T0*(
_output_shapes
:         АТ
update_connection/mul_19Mul update_connection/Sigmoid_12:y:0update_connection/add_23:z:0*
T0*(
_output_shapes
:         А^
update_connection/sub_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ч
update_connection/sub_6Sub"update_connection/sub_6/x:output:0 update_connection/Sigmoid_12:y:0*
T0*(
_output_shapes
:         АН
update_connection/mul_20Mulupdate_connection/sub_6:z:0update_connection/Tanh_6:y:0*
T0*(
_output_shapes
:         АР
update_connection/add_27AddV2update_connection/mul_19:z:0update_connection/mul_20:z:0*
T0*(
_output_shapes
:         АR
GatherV2_28/axisConst*
_output_shapes
: *
dtype0*
value	B : в
GatherV2_28GatherV2update_ip/add_27:z:0Squeeze_1:output:0GatherV2_28/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_29/axisConst*
_output_shapes
: *
dtype0*
value	B : к
GatherV2_29GatherV2update_connection/add_27:z:0Squeeze_2:output:0GatherV2_29/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_33SqueezeGatherV2_29:output:0*
T0*
_output_shapes
:N

Squeeze_34SqueezeGatherV2_28:output:0*
T0*
_output_shapes
:P
concat_15/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_15ConcatV2Squeeze_34:output:0Squeeze_33:output:0concat_15/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_28EnsureShapeconcat_15:output:0*
T0*(
_output_shapes
:         А*
shape:         Аu
sequential/dropout/Identity_7IdentityEnsureShape_28:output:0*
T0*(
_output_shapes
:         АЪ
(sequential/dense/MatMul_7/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0░
sequential/dense/MatMul_7MatMul&sequential/dropout/Identity_7:output:00sequential/dense/MatMul_7/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЧ
)sequential/dense/BiasAdd_7/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0░
sequential/dense/BiasAdd_7BiasAdd#sequential/dense/MatMul_7:product:01sequential/dense/BiasAdd_7/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аw
sequential/dense/Relu_7Relu#sequential/dense/BiasAdd_7:output:0*
T0*(
_output_shapes
:         Аu
UnsortedSegmentMean_14/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧f
!UnsortedSegmentMean_14/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
UnsortedSegmentMean_14/onesFill%UnsortedSegmentMean_14/Shape:output:0*UnsortedSegmentMean_14/ones/Const:output:0*
T0*
_output_shapes
:╩
)UnsortedSegmentMean_14/UnsortedSegmentSumUnsortedSegmentSum$UnsortedSegmentMean_14/ones:output:0Squeeze_2:output:0inputs_2*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:t
*UnsortedSegmentMean_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:и
$UnsortedSegmentMean_14/strided_sliceStridedSliceinputs_23UnsortedSegmentMean_14/strided_slice/stack:output:05UnsortedSegmentMean_14/strided_slice/stack_1:output:05UnsortedSegmentMean_14/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask]
UnsortedSegmentMean_14/RankConst*
_output_shapes
: *
dtype0*
value	B :Z
UnsortedSegmentMean_14/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Р
UnsortedSegmentMean_14/subSub$UnsortedSegmentMean_14/Rank:output:0&UnsortedSegmentMean_14/Rank_1:output:0*
T0*
_output_shapes
: z
$UnsortedSegmentMean_14/ones_1/packedPackUnsortedSegmentMean_14/sub:z:0*
N*
T0*
_output_shapes
:e
#UnsortedSegmentMean_14/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R░
UnsortedSegmentMean_14/ones_1Fill-UnsortedSegmentMean_14/ones_1/packed:output:0,UnsortedSegmentMean_14/ones_1/Const:output:0*
T0	*#
_output_shapes
:         d
"UnsortedSegmentMean_14/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
UnsortedSegmentMean_14/concatConcatV2-UnsortedSegmentMean_14/strided_slice:output:0&UnsortedSegmentMean_14/ones_1:output:0+UnsortedSegmentMean_14/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         ╢
UnsortedSegmentMean_14/ReshapeReshape2UnsortedSegmentMean_14/UnsortedSegmentSum:output:0&UnsortedSegmentMean_14/concat:output:0*
Tshape0	*
T0*
_output_shapes
:e
 UnsortedSegmentMean_14/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
UnsortedSegmentMean_14/MaximumMaximum'UnsortedSegmentMean_14/Reshape:output:0)UnsortedSegmentMean_14/Maximum/y:output:0*
T0*
_output_shapes
:═
+UnsortedSegmentMean_14/UnsortedSegmentSum_1UnsortedSegmentSum%sequential/dense/Relu_7:activations:0Squeeze_2:output:0inputs_2*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:ж
UnsortedSegmentMean_14/truedivRealDiv4UnsortedSegmentMean_14/UnsortedSegmentSum_1:output:0"UnsortedSegmentMean_14/Maximum:z:0*
T0*
_output_shapes
:R
GatherV2_30/axisConst*
_output_shapes
: *
dtype0*
value	B : к
GatherV2_30GatherV2update_connection/add_27:z:0Squeeze_3:output:0GatherV2_30/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_31/axisConst*
_output_shapes
: *
dtype0*
value	B : в
GatherV2_31GatherV2update_ip/add_27:z:0Squeeze_4:output:0GatherV2_31/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_35SqueezeGatherV2_31:output:0*
T0*
_output_shapes
:N

Squeeze_36SqueezeGatherV2_30:output:0*
T0*
_output_shapes
:P
concat_16/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_16ConcatV2Squeeze_36:output:0Squeeze_35:output:0concat_16/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_29EnsureShapeconcat_16:output:0*
T0*(
_output_shapes
:         А*
shape:         Аy
!sequential_1/dropout_1/Identity_7IdentityEnsureShape_29:output:0*
T0*(
_output_shapes
:         Ав
,sequential_1/dense_1/MatMul_7/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╝
sequential_1/dense_1/MatMul_7MatMul*sequential_1/dropout_1/Identity_7:output:04sequential_1/dense_1/MatMul_7/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
-sequential_1/dense_1/BiasAdd_7/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╝
sequential_1/dense_1/BiasAdd_7BiasAdd'sequential_1/dense_1/MatMul_7:product:05sequential_1/dense_1/BiasAdd_7/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А
sequential_1/dense_1/Relu_7Relu'sequential_1/dense_1/BiasAdd_7:output:0*
T0*(
_output_shapes
:         Аu
UnsortedSegmentMean_15/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧f
!UnsortedSegmentMean_15/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
UnsortedSegmentMean_15/onesFill%UnsortedSegmentMean_15/Shape:output:0*UnsortedSegmentMean_15/ones/Const:output:0*
T0*
_output_shapes
:╩
)UnsortedSegmentMean_15/UnsortedSegmentSumUnsortedSegmentSum$UnsortedSegmentMean_15/ones:output:0Squeeze_4:output:0inputs_1*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:t
*UnsortedSegmentMean_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:и
$UnsortedSegmentMean_15/strided_sliceStridedSliceinputs_13UnsortedSegmentMean_15/strided_slice/stack:output:05UnsortedSegmentMean_15/strided_slice/stack_1:output:05UnsortedSegmentMean_15/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask]
UnsortedSegmentMean_15/RankConst*
_output_shapes
: *
dtype0*
value	B :Z
UnsortedSegmentMean_15/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Р
UnsortedSegmentMean_15/subSub$UnsortedSegmentMean_15/Rank:output:0&UnsortedSegmentMean_15/Rank_1:output:0*
T0*
_output_shapes
: z
$UnsortedSegmentMean_15/ones_1/packedPackUnsortedSegmentMean_15/sub:z:0*
N*
T0*
_output_shapes
:e
#UnsortedSegmentMean_15/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R░
UnsortedSegmentMean_15/ones_1Fill-UnsortedSegmentMean_15/ones_1/packed:output:0,UnsortedSegmentMean_15/ones_1/Const:output:0*
T0	*#
_output_shapes
:         d
"UnsortedSegmentMean_15/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
UnsortedSegmentMean_15/concatConcatV2-UnsortedSegmentMean_15/strided_slice:output:0&UnsortedSegmentMean_15/ones_1:output:0+UnsortedSegmentMean_15/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         ╢
UnsortedSegmentMean_15/ReshapeReshape2UnsortedSegmentMean_15/UnsortedSegmentSum:output:0&UnsortedSegmentMean_15/concat:output:0*
Tshape0	*
T0*
_output_shapes
:e
 UnsortedSegmentMean_15/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
UnsortedSegmentMean_15/MaximumMaximum'UnsortedSegmentMean_15/Reshape:output:0)UnsortedSegmentMean_15/Maximum/y:output:0*
T0*
_output_shapes
:╤
+UnsortedSegmentMean_15/UnsortedSegmentSum_1UnsortedSegmentSum)sequential_1/dense_1/Relu_7:activations:0Squeeze_4:output:0inputs_1*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:ж
UnsortedSegmentMean_15/truedivRealDiv4UnsortedSegmentMean_15/UnsortedSegmentSum_1:output:0"UnsortedSegmentMean_15/Maximum:z:0*
T0*
_output_shapes
:У
EnsureShape_30EnsureShape"UnsortedSegmentMean_15/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         А}
update_ip/ReadVariableOp_7ReadVariableOp!update_ip_readvariableop_resource*
_output_shapes
:	А*
dtype0y
update_ip/unstack_7Unpack"update_ip/ReadVariableOp_7:value:0*
T0*"
_output_shapes
:А:А*	
numН
"update_ip/MatMul_14/ReadVariableOpReadVariableOp(update_ip_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Х
update_ip/MatMul_14MatMulEnsureShape_30:output:0*update_ip/MatMul_14/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АП
update_ip/BiasAdd_14BiasAddupdate_ip/MatMul_14:product:0update_ip/unstack_7:output:0*
T0*(
_output_shapes
:         Аg
update_ip/split_14/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ═
update_ip/split_14Split%update_ip/split_14/split_dim:output:0update_ip/BiasAdd_14:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitП
"update_ip/MatMul_15/ReadVariableOpReadVariableOp*update_ip_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Т
update_ip/MatMul_15MatMulupdate_ip/add_27:z:0*update_ip/MatMul_15/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АП
update_ip/BiasAdd_15BiasAddupdate_ip/MatMul_15:product:0update_ip/unstack_7:output:1*
T0*(
_output_shapes
:         Аf
update_ip/Const_7Const*
_output_shapes
:*
dtype0*!
valueB"А   А       g
update_ip/split_15/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Ў
update_ip/split_15SplitVupdate_ip/BiasAdd_15:output:0update_ip/Const_7:output:0%update_ip/split_15/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЖ
update_ip/add_28AddV2update_ip/split_14:output:0update_ip/split_15:output:0*
T0*(
_output_shapes
:         Аh
update_ip/Sigmoid_14Sigmoidupdate_ip/add_28:z:0*
T0*(
_output_shapes
:         АЖ
update_ip/add_29AddV2update_ip/split_14:output:1update_ip/split_15:output:1*
T0*(
_output_shapes
:         Аh
update_ip/Sigmoid_15Sigmoidupdate_ip/add_29:z:0*
T0*(
_output_shapes
:         АБ
update_ip/mul_21Mulupdate_ip/Sigmoid_15:y:0update_ip/split_15:output:2*
T0*(
_output_shapes
:         А
update_ip/add_30AddV2update_ip/split_14:output:2update_ip/mul_21:z:0*
T0*(
_output_shapes
:         Аa
update_ip/Tanh_7Tanhupdate_ip/add_30:z:0*
T0*(
_output_shapes
:         Аz
update_ip/mul_22Mulupdate_ip/Sigmoid_14:y:0update_ip/add_27:z:0*
T0*(
_output_shapes
:         АV
update_ip/sub_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
update_ip/sub_7Subupdate_ip/sub_7/x:output:0update_ip/Sigmoid_14:y:0*
T0*(
_output_shapes
:         Аu
update_ip/mul_23Mulupdate_ip/sub_7:z:0update_ip/Tanh_7:y:0*
T0*(
_output_shapes
:         Аx
update_ip/add_31AddV2update_ip/mul_22:z:0update_ip/mul_23:z:0*
T0*(
_output_shapes
:         АУ
EnsureShape_31EnsureShape"UnsortedSegmentMean_14/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АН
"update_connection/ReadVariableOp_7ReadVariableOp)update_connection_readvariableop_resource*
_output_shapes
:	А*
dtype0Й
update_connection/unstack_7Unpack*update_connection/ReadVariableOp_7:value:0*
T0*"
_output_shapes
:А:А*	
numЭ
*update_connection/MatMul_14/ReadVariableOpReadVariableOp0update_connection_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0е
update_connection/MatMul_14MatMulEnsureShape_31:output:02update_connection/MatMul_14/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аз
update_connection/BiasAdd_14BiasAdd%update_connection/MatMul_14:product:0$update_connection/unstack_7:output:0*
T0*(
_output_shapes
:         Аo
$update_connection/split_14/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         х
update_connection/split_14Split-update_connection/split_14/split_dim:output:0%update_connection/BiasAdd_14:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЯ
*update_connection/MatMul_15/ReadVariableOpReadVariableOp2update_connection_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0к
update_connection/MatMul_15MatMulupdate_connection/add_27:z:02update_connection/MatMul_15/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аз
update_connection/BiasAdd_15BiasAdd%update_connection/MatMul_15:product:0$update_connection/unstack_7:output:1*
T0*(
_output_shapes
:         Аn
update_connection/Const_7Const*
_output_shapes
:*
dtype0*!
valueB"А   А       o
$update_connection/split_15/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Ц
update_connection/split_15SplitV%update_connection/BiasAdd_15:output:0"update_connection/Const_7:output:0-update_connection/split_15/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЮ
update_connection/add_28AddV2#update_connection/split_14:output:0#update_connection/split_15:output:0*
T0*(
_output_shapes
:         Аx
update_connection/Sigmoid_14Sigmoidupdate_connection/add_28:z:0*
T0*(
_output_shapes
:         АЮ
update_connection/add_29AddV2#update_connection/split_14:output:1#update_connection/split_15:output:1*
T0*(
_output_shapes
:         Аx
update_connection/Sigmoid_15Sigmoidupdate_connection/add_29:z:0*
T0*(
_output_shapes
:         АЩ
update_connection/mul_21Mul update_connection/Sigmoid_15:y:0#update_connection/split_15:output:2*
T0*(
_output_shapes
:         АЧ
update_connection/add_30AddV2#update_connection/split_14:output:2update_connection/mul_21:z:0*
T0*(
_output_shapes
:         Аq
update_connection/Tanh_7Tanhupdate_connection/add_30:z:0*
T0*(
_output_shapes
:         АТ
update_connection/mul_22Mul update_connection/Sigmoid_14:y:0update_connection/add_27:z:0*
T0*(
_output_shapes
:         А^
update_connection/sub_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ч
update_connection/sub_7Sub"update_connection/sub_7/x:output:0 update_connection/Sigmoid_14:y:0*
T0*(
_output_shapes
:         АН
update_connection/mul_23Mulupdate_connection/sub_7:z:0update_connection/Tanh_7:y:0*
T0*(
_output_shapes
:         АР
update_connection/add_31AddV2update_connection/mul_22:z:0update_connection/mul_23:z:0*
T0*(
_output_shapes
:         Аа
*sequential_2/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0к
sequential_2/dense_2/MatMulMatMulupdate_connection/add_31:z:02sequential_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЭ
+sequential_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╢
sequential_2/dense_2/BiasAddBiasAdd%sequential_2/dense_2/MatMul:product:03sequential_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А{
sequential_2/dense_2/ReluRelu%sequential_2/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:         АЗ
sequential_2/dropout_2/IdentityIdentity'sequential_2/dense_2/Relu:activations:0*
T0*(
_output_shapes
:         АЯ
*sequential_2/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_3_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0╡
sequential_2/dense_3/MatMulMatMul(sequential_2/dropout_2/Identity:output:02sequential_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ь
+sequential_2/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╡
sequential_2/dense_3/BiasAddBiasAdd%sequential_2/dense_3/MatMul:product:03sequential_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @z
sequential_2/dense_3/ReluRelu%sequential_2/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:         @Ж
sequential_2/dropout_3/IdentityIdentity'sequential_2/dense_3/Relu:activations:0*
T0*'
_output_shapes
:         @Ю
*sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0╡
sequential_2/dense_4/MatMulMatMul(sequential_2/dropout_3/Identity:output:02sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ь
+sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╡
sequential_2/dense_4/BiasAddBiasAdd%sequential_2/dense_4/MatMul:product:03sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         А
sequential_2/dense_4/SoftmaxSoftmax%sequential_2/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:         u
IdentityIdentity&sequential_2/dense_4/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         я
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/BiasAdd_1/ReadVariableOp*^sequential/dense/BiasAdd_2/ReadVariableOp*^sequential/dense/BiasAdd_3/ReadVariableOp*^sequential/dense/BiasAdd_4/ReadVariableOp*^sequential/dense/BiasAdd_5/ReadVariableOp*^sequential/dense/BiasAdd_6/ReadVariableOp*^sequential/dense/BiasAdd_7/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp)^sequential/dense/MatMul_1/ReadVariableOp)^sequential/dense/MatMul_2/ReadVariableOp)^sequential/dense/MatMul_3/ReadVariableOp)^sequential/dense/MatMul_4/ReadVariableOp)^sequential/dense/MatMul_5/ReadVariableOp)^sequential/dense/MatMul_6/ReadVariableOp)^sequential/dense/MatMul_7/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp.^sequential_1/dense_1/BiasAdd_1/ReadVariableOp.^sequential_1/dense_1/BiasAdd_2/ReadVariableOp.^sequential_1/dense_1/BiasAdd_3/ReadVariableOp.^sequential_1/dense_1/BiasAdd_4/ReadVariableOp.^sequential_1/dense_1/BiasAdd_5/ReadVariableOp.^sequential_1/dense_1/BiasAdd_6/ReadVariableOp.^sequential_1/dense_1/BiasAdd_7/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp-^sequential_1/dense_1/MatMul_1/ReadVariableOp-^sequential_1/dense_1/MatMul_2/ReadVariableOp-^sequential_1/dense_1/MatMul_3/ReadVariableOp-^sequential_1/dense_1/MatMul_4/ReadVariableOp-^sequential_1/dense_1/MatMul_5/ReadVariableOp-^sequential_1/dense_1/MatMul_6/ReadVariableOp-^sequential_1/dense_1/MatMul_7/ReadVariableOp,^sequential_2/dense_2/BiasAdd/ReadVariableOp+^sequential_2/dense_2/MatMul/ReadVariableOp,^sequential_2/dense_3/BiasAdd/ReadVariableOp+^sequential_2/dense_3/MatMul/ReadVariableOp,^sequential_2/dense_4/BiasAdd/ReadVariableOp+^sequential_2/dense_4/MatMul/ReadVariableOp(^update_connection/MatMul/ReadVariableOp*^update_connection/MatMul_1/ReadVariableOp+^update_connection/MatMul_10/ReadVariableOp+^update_connection/MatMul_11/ReadVariableOp+^update_connection/MatMul_12/ReadVariableOp+^update_connection/MatMul_13/ReadVariableOp+^update_connection/MatMul_14/ReadVariableOp+^update_connection/MatMul_15/ReadVariableOp*^update_connection/MatMul_2/ReadVariableOp*^update_connection/MatMul_3/ReadVariableOp*^update_connection/MatMul_4/ReadVariableOp*^update_connection/MatMul_5/ReadVariableOp*^update_connection/MatMul_6/ReadVariableOp*^update_connection/MatMul_7/ReadVariableOp*^update_connection/MatMul_8/ReadVariableOp*^update_connection/MatMul_9/ReadVariableOp!^update_connection/ReadVariableOp#^update_connection/ReadVariableOp_1#^update_connection/ReadVariableOp_2#^update_connection/ReadVariableOp_3#^update_connection/ReadVariableOp_4#^update_connection/ReadVariableOp_5#^update_connection/ReadVariableOp_6#^update_connection/ReadVariableOp_7 ^update_ip/MatMul/ReadVariableOp"^update_ip/MatMul_1/ReadVariableOp#^update_ip/MatMul_10/ReadVariableOp#^update_ip/MatMul_11/ReadVariableOp#^update_ip/MatMul_12/ReadVariableOp#^update_ip/MatMul_13/ReadVariableOp#^update_ip/MatMul_14/ReadVariableOp#^update_ip/MatMul_15/ReadVariableOp"^update_ip/MatMul_2/ReadVariableOp"^update_ip/MatMul_3/ReadVariableOp"^update_ip/MatMul_4/ReadVariableOp"^update_ip/MatMul_5/ReadVariableOp"^update_ip/MatMul_6/ReadVariableOp"^update_ip/MatMul_7/ReadVariableOp"^update_ip/MatMul_8/ReadVariableOp"^update_ip/MatMul_9/ReadVariableOp^update_ip/ReadVariableOp^update_ip/ReadVariableOp_1^update_ip/ReadVariableOp_2^update_ip/ReadVariableOp_3^update_ip/ReadVariableOp_4^update_ip/ReadVariableOp_5^update_ip/ReadVariableOp_6^update_ip/ReadVariableOp_7*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:::: : ::: : : : : : : : : : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/BiasAdd_1/ReadVariableOp)sequential/dense/BiasAdd_1/ReadVariableOp2V
)sequential/dense/BiasAdd_2/ReadVariableOp)sequential/dense/BiasAdd_2/ReadVariableOp2V
)sequential/dense/BiasAdd_3/ReadVariableOp)sequential/dense/BiasAdd_3/ReadVariableOp2V
)sequential/dense/BiasAdd_4/ReadVariableOp)sequential/dense/BiasAdd_4/ReadVariableOp2V
)sequential/dense/BiasAdd_5/ReadVariableOp)sequential/dense/BiasAdd_5/ReadVariableOp2V
)sequential/dense/BiasAdd_6/ReadVariableOp)sequential/dense/BiasAdd_6/ReadVariableOp2V
)sequential/dense/BiasAdd_7/ReadVariableOp)sequential/dense/BiasAdd_7/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2T
(sequential/dense/MatMul_1/ReadVariableOp(sequential/dense/MatMul_1/ReadVariableOp2T
(sequential/dense/MatMul_2/ReadVariableOp(sequential/dense/MatMul_2/ReadVariableOp2T
(sequential/dense/MatMul_3/ReadVariableOp(sequential/dense/MatMul_3/ReadVariableOp2T
(sequential/dense/MatMul_4/ReadVariableOp(sequential/dense/MatMul_4/ReadVariableOp2T
(sequential/dense/MatMul_5/ReadVariableOp(sequential/dense/MatMul_5/ReadVariableOp2T
(sequential/dense/MatMul_6/ReadVariableOp(sequential/dense/MatMul_6/ReadVariableOp2T
(sequential/dense/MatMul_7/ReadVariableOp(sequential/dense/MatMul_7/ReadVariableOp2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2^
-sequential_1/dense_1/BiasAdd_1/ReadVariableOp-sequential_1/dense_1/BiasAdd_1/ReadVariableOp2^
-sequential_1/dense_1/BiasAdd_2/ReadVariableOp-sequential_1/dense_1/BiasAdd_2/ReadVariableOp2^
-sequential_1/dense_1/BiasAdd_3/ReadVariableOp-sequential_1/dense_1/BiasAdd_3/ReadVariableOp2^
-sequential_1/dense_1/BiasAdd_4/ReadVariableOp-sequential_1/dense_1/BiasAdd_4/ReadVariableOp2^
-sequential_1/dense_1/BiasAdd_5/ReadVariableOp-sequential_1/dense_1/BiasAdd_5/ReadVariableOp2^
-sequential_1/dense_1/BiasAdd_6/ReadVariableOp-sequential_1/dense_1/BiasAdd_6/ReadVariableOp2^
-sequential_1/dense_1/BiasAdd_7/ReadVariableOp-sequential_1/dense_1/BiasAdd_7/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp2\
,sequential_1/dense_1/MatMul_1/ReadVariableOp,sequential_1/dense_1/MatMul_1/ReadVariableOp2\
,sequential_1/dense_1/MatMul_2/ReadVariableOp,sequential_1/dense_1/MatMul_2/ReadVariableOp2\
,sequential_1/dense_1/MatMul_3/ReadVariableOp,sequential_1/dense_1/MatMul_3/ReadVariableOp2\
,sequential_1/dense_1/MatMul_4/ReadVariableOp,sequential_1/dense_1/MatMul_4/ReadVariableOp2\
,sequential_1/dense_1/MatMul_5/ReadVariableOp,sequential_1/dense_1/MatMul_5/ReadVariableOp2\
,sequential_1/dense_1/MatMul_6/ReadVariableOp,sequential_1/dense_1/MatMul_6/ReadVariableOp2\
,sequential_1/dense_1/MatMul_7/ReadVariableOp,sequential_1/dense_1/MatMul_7/ReadVariableOp2Z
+sequential_2/dense_2/BiasAdd/ReadVariableOp+sequential_2/dense_2/BiasAdd/ReadVariableOp2X
*sequential_2/dense_2/MatMul/ReadVariableOp*sequential_2/dense_2/MatMul/ReadVariableOp2Z
+sequential_2/dense_3/BiasAdd/ReadVariableOp+sequential_2/dense_3/BiasAdd/ReadVariableOp2X
*sequential_2/dense_3/MatMul/ReadVariableOp*sequential_2/dense_3/MatMul/ReadVariableOp2Z
+sequential_2/dense_4/BiasAdd/ReadVariableOp+sequential_2/dense_4/BiasAdd/ReadVariableOp2X
*sequential_2/dense_4/MatMul/ReadVariableOp*sequential_2/dense_4/MatMul/ReadVariableOp2R
'update_connection/MatMul/ReadVariableOp'update_connection/MatMul/ReadVariableOp2V
)update_connection/MatMul_1/ReadVariableOp)update_connection/MatMul_1/ReadVariableOp2X
*update_connection/MatMul_10/ReadVariableOp*update_connection/MatMul_10/ReadVariableOp2X
*update_connection/MatMul_11/ReadVariableOp*update_connection/MatMul_11/ReadVariableOp2X
*update_connection/MatMul_12/ReadVariableOp*update_connection/MatMul_12/ReadVariableOp2X
*update_connection/MatMul_13/ReadVariableOp*update_connection/MatMul_13/ReadVariableOp2X
*update_connection/MatMul_14/ReadVariableOp*update_connection/MatMul_14/ReadVariableOp2X
*update_connection/MatMul_15/ReadVariableOp*update_connection/MatMul_15/ReadVariableOp2V
)update_connection/MatMul_2/ReadVariableOp)update_connection/MatMul_2/ReadVariableOp2V
)update_connection/MatMul_3/ReadVariableOp)update_connection/MatMul_3/ReadVariableOp2V
)update_connection/MatMul_4/ReadVariableOp)update_connection/MatMul_4/ReadVariableOp2V
)update_connection/MatMul_5/ReadVariableOp)update_connection/MatMul_5/ReadVariableOp2V
)update_connection/MatMul_6/ReadVariableOp)update_connection/MatMul_6/ReadVariableOp2V
)update_connection/MatMul_7/ReadVariableOp)update_connection/MatMul_7/ReadVariableOp2V
)update_connection/MatMul_8/ReadVariableOp)update_connection/MatMul_8/ReadVariableOp2V
)update_connection/MatMul_9/ReadVariableOp)update_connection/MatMul_9/ReadVariableOp2H
"update_connection/ReadVariableOp_1"update_connection/ReadVariableOp_12H
"update_connection/ReadVariableOp_2"update_connection/ReadVariableOp_22H
"update_connection/ReadVariableOp_3"update_connection/ReadVariableOp_32H
"update_connection/ReadVariableOp_4"update_connection/ReadVariableOp_42H
"update_connection/ReadVariableOp_5"update_connection/ReadVariableOp_52H
"update_connection/ReadVariableOp_6"update_connection/ReadVariableOp_62H
"update_connection/ReadVariableOp_7"update_connection/ReadVariableOp_72D
 update_connection/ReadVariableOp update_connection/ReadVariableOp2B
update_ip/MatMul/ReadVariableOpupdate_ip/MatMul/ReadVariableOp2F
!update_ip/MatMul_1/ReadVariableOp!update_ip/MatMul_1/ReadVariableOp2H
"update_ip/MatMul_10/ReadVariableOp"update_ip/MatMul_10/ReadVariableOp2H
"update_ip/MatMul_11/ReadVariableOp"update_ip/MatMul_11/ReadVariableOp2H
"update_ip/MatMul_12/ReadVariableOp"update_ip/MatMul_12/ReadVariableOp2H
"update_ip/MatMul_13/ReadVariableOp"update_ip/MatMul_13/ReadVariableOp2H
"update_ip/MatMul_14/ReadVariableOp"update_ip/MatMul_14/ReadVariableOp2H
"update_ip/MatMul_15/ReadVariableOp"update_ip/MatMul_15/ReadVariableOp2F
!update_ip/MatMul_2/ReadVariableOp!update_ip/MatMul_2/ReadVariableOp2F
!update_ip/MatMul_3/ReadVariableOp!update_ip/MatMul_3/ReadVariableOp2F
!update_ip/MatMul_4/ReadVariableOp!update_ip/MatMul_4/ReadVariableOp2F
!update_ip/MatMul_5/ReadVariableOp!update_ip/MatMul_5/ReadVariableOp2F
!update_ip/MatMul_6/ReadVariableOp!update_ip/MatMul_6/ReadVariableOp2F
!update_ip/MatMul_7/ReadVariableOp!update_ip/MatMul_7/ReadVariableOp2F
!update_ip/MatMul_8/ReadVariableOp!update_ip/MatMul_8/ReadVariableOp2F
!update_ip/MatMul_9/ReadVariableOp!update_ip/MatMul_9/ReadVariableOp28
update_ip/ReadVariableOp_1update_ip/ReadVariableOp_128
update_ip/ReadVariableOp_2update_ip/ReadVariableOp_228
update_ip/ReadVariableOp_3update_ip/ReadVariableOp_328
update_ip/ReadVariableOp_4update_ip/ReadVariableOp_428
update_ip/ReadVariableOp_5update_ip/ReadVariableOp_528
update_ip/ReadVariableOp_6update_ip/ReadVariableOp_628
update_ip/ReadVariableOp_7update_ip/ReadVariableOp_724
update_ip/ReadVariableOpupdate_ip/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:@<

_output_shapes
:
 
_user_specified_nameinputs:@<

_output_shapes
:
 
_user_specified_nameinputs:>:

_output_shapes
: 
 
_user_specified_nameinputs:>:

_output_shapes
: 
 
_user_specified_nameinputs:@<

_output_shapes
:
 
_user_specified_nameinputs:@<

_output_shapes
:
 
_user_specified_nameinputs:@ <

_output_shapes
:
 
_user_specified_nameinputs
є
Ч
'__inference_dense_layer_call_fn_4847704

inputs
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCall╪
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_4845434p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	4847700:'#
!
_user_specified_name	4847698:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
б
E
)__inference_dropout_layer_call_fn_4847678

inputs
identity░
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_4845448a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┘
d
F__inference_dropout_3_layer_call_and_return_conditional_losses_4847856

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         @[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
┴
╨
I__inference_sequential_1_layer_call_and_return_conditional_losses_4845541
input_2#
dense_1_4845535:
АА
dense_1_4845537:	А
identityИвdense_1/StatefulPartitionedCall╜
dropout_1/PartitionedCallPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_4845533М
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_4845535dense_1_4845537*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_4845519x
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         АD
NoOpNoOp ^dense_1/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:'#
!
_user_specified_name	4845537:'#
!
_user_specified_name	4845535:Q M
(
_output_shapes
:         А
!
_user_specified_name	input_2
Л
т
N__inference_update_connection_layer_call_and_return_conditional_losses_4845950

inputs

states*
readvariableop_resource:	А2
matmul_readvariableop_resource:
АА4
 matmul_1_readvariableop_resource:
АА
identity

identity_1ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	А*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:А:А*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         АZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         АZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"А   А       \
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╔
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         АN
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         Аc
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         АR
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         А^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         АZ
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         АJ
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:         АT
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:         АJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         АR
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:         АW
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         АY
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         А[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         Аe
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         А:                  : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:XT
0
_output_shapes
:                  
 
_user_specified_namestates:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╧
b
)__inference_dropout_layer_call_fn_4847673

inputs
identityИвStatefulPartitionedCall└
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_4845422p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╒

Ў
B__inference_dense_layer_call_and_return_conditional_losses_4847715

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         АS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
У
▄
+__inference_update_ip_layer_call_fn_4847470

inputs
states_0
unknown:	А
	unknown_0:
АА
	unknown_1:
АА
identity

identity_1ИвStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_update_ip_layer_call_and_return_conditional_losses_4845902p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         Аr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         А:         А: : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	4847464:'#
!
_user_specified_name	4847462:'#
!
_user_specified_name	4847460:RN
(
_output_shapes
:         А
"
_user_specified_name
states_0:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╙
d
+__inference_dropout_2_layer_call_fn_4847787

inputs
identityИвStatefulPartitionedCall┬
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_4845608p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
│
ф
3__inference_update_connection_layer_call_fn_4847590

inputs
states_0
unknown:	А
	unknown_0:
АА
	unknown_1:
АА
identity

identity_1ИвStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_update_connection_layer_call_and_return_conditional_losses_4846701p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         Аr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         А:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	4847584:'#
!
_user_specified_name	4847582:'#
!
_user_specified_name	4847580:ZV
0
_output_shapes
:                  
"
_user_specified_name
states_0:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╩╫
ж
@__inference_gnn_layer_call_and_return_conditional_losses_4847270
dst_connection_to_ip	
dst_ip_to_connection	
feature_connection
n_c	
n_i	
src_connection_to_ip	
src_ip_to_connection	&
sequential_4846553:
АА!
sequential_4846555:	А(
sequential_1_4846588:
АА#
sequential_1_4846590:	А$
update_ip_4846654:	А%
update_ip_4846656:
АА%
update_ip_4846658:
АА,
update_connection_4846702:	А-
update_connection_4846704:
АА-
update_connection_4846706:
АА(
sequential_2_4847256:
АА#
sequential_2_4847258:	А'
sequential_2_4847260:	А@"
sequential_2_4847262:@&
sequential_2_4847264:@"
sequential_2_4847266:
identityИв"sequential/StatefulPartitionedCallв$sequential/StatefulPartitionedCall_1в$sequential/StatefulPartitionedCall_2в$sequential/StatefulPartitionedCall_3в$sequential/StatefulPartitionedCall_4в$sequential/StatefulPartitionedCall_5в$sequential/StatefulPartitionedCall_6в$sequential/StatefulPartitionedCall_7в$sequential_1/StatefulPartitionedCallв&sequential_1/StatefulPartitionedCall_1в&sequential_1/StatefulPartitionedCall_2в&sequential_1/StatefulPartitionedCall_3в&sequential_1/StatefulPartitionedCall_4в&sequential_1/StatefulPartitionedCall_5в&sequential_1/StatefulPartitionedCall_6в&sequential_1/StatefulPartitionedCall_7в$sequential_2/StatefulPartitionedCallв)update_connection/StatefulPartitionedCallв+update_connection/StatefulPartitionedCall_1в+update_connection/StatefulPartitionedCall_2в+update_connection/StatefulPartitionedCall_3в+update_connection/StatefulPartitionedCall_4в+update_connection/StatefulPartitionedCall_5в+update_connection/StatefulPartitionedCall_6в+update_connection/StatefulPartitionedCall_7в!update_ip/StatefulPartitionedCallв#update_ip/StatefulPartitionedCall_1в#update_ip/StatefulPartitionedCall_2в#update_ip/StatefulPartitionedCall_3в#update_ip/StatefulPartitionedCall_4в#update_ip/StatefulPartitionedCall_5в#update_ip/StatefulPartitionedCall_6в#update_ip/StatefulPartitionedCall_7I
SqueezeSqueezefeature_connection*
T0*
_output_shapes
:M
	Squeeze_1Squeezesrc_ip_to_connection*
T0	*
_output_shapes
:M
	Squeeze_2Squeezedst_ip_to_connection*
T0	*
_output_shapes
:M
	Squeeze_3Squeezesrc_connection_to_ip*
T0	*
_output_shapes
:M
	Squeeze_4Squeezedst_connection_to_ip*
T0	*
_output_shapes
:F
	ones/CastCastn_i*

DstT0*

SrcT0	*
_output_shapes
: P
ones/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Аh
ones/packedPackones/Cast:y:0ones/packed/1:output:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?j
onesFillones/packed:output:0ones/Const:output:0*
T0*(
_output_shapes
:         АI
stack/1Const*
_output_shapes
: *
dtype0	*
value	B	 RfR
stackPackn_cstack/1:output:0*
N*
T0	*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    w
zerosFillstack:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         f*

index_type0	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :О
concatConcatV2Squeeze:output:0zeros:output:0concat/axis:output:0*
N*
T0*0
_output_shapes
:                  O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Х
GatherV2GatherV2ones:output:0Squeeze_1:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:Q
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ы

GatherV2_1GatherV2concat:output:0Squeeze_2:output:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:L
	Squeeze_5SqueezeGatherV2_1:output:0*
T0*
_output_shapes
:J
	Squeeze_6SqueezeGatherV2:output:0*
T0*
_output_shapes
:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :А
concat_1ConcatV2Squeeze_6:output:0Squeeze_5:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:
EnsureShapeEnsureShapeconcat_1:output:0*
T0*(
_output_shapes
:         А*
shape:         АК
"sequential/StatefulPartitionedCallStatefulPartitionedCallEnsureShape:output:0sequential_4846553sequential_4846555*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4845456r
UnsortedSegmentMean/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧c
UnsortedSegmentMean/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Р
UnsortedSegmentMean/onesFill"UnsortedSegmentMean/Shape:output:0'UnsortedSegmentMean/ones/Const:output:0*
T0*
_output_shapes
:┐
&UnsortedSegmentMean/UnsortedSegmentSumUnsortedSegmentSum!UnsortedSegmentMean/ones:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:q
'UnsortedSegmentMean/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)UnsortedSegmentMean/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: s
)UnsortedSegmentMean/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
!UnsortedSegmentMean/strided_sliceStridedSlicen_c0UnsortedSegmentMean/strided_slice/stack:output:02UnsortedSegmentMean/strided_slice/stack_1:output:02UnsortedSegmentMean/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_maskZ
UnsortedSegmentMean/RankConst*
_output_shapes
: *
dtype0*
value	B :W
UnsortedSegmentMean/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: З
UnsortedSegmentMean/subSub!UnsortedSegmentMean/Rank:output:0#UnsortedSegmentMean/Rank_1:output:0*
T0*
_output_shapes
: t
!UnsortedSegmentMean/ones_1/packedPackUnsortedSegmentMean/sub:z:0*
N*
T0*
_output_shapes
:b
 UnsortedSegmentMean/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rз
UnsortedSegmentMean/ones_1Fill*UnsortedSegmentMean/ones_1/packed:output:0)UnsortedSegmentMean/ones_1/Const:output:0*
T0	*#
_output_shapes
:         a
UnsortedSegmentMean/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╪
UnsortedSegmentMean/concatConcatV2*UnsortedSegmentMean/strided_slice:output:0#UnsortedSegmentMean/ones_1:output:0(UnsortedSegmentMean/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         н
UnsortedSegmentMean/ReshapeReshape/UnsortedSegmentMean/UnsortedSegmentSum:output:0#UnsortedSegmentMean/concat:output:0*
Tshape0	*
T0*
_output_shapes
:b
UnsortedSegmentMean/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ч
UnsortedSegmentMean/MaximumMaximum$UnsortedSegmentMean/Reshape:output:0&UnsortedSegmentMean/Maximum/y:output:0*
T0*
_output_shapes
:╦
(UnsortedSegmentMean/UnsortedSegmentSum_1UnsortedSegmentSum+sequential/StatefulPartitionedCall:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:Э
UnsortedSegmentMean/truedivRealDiv1UnsortedSegmentMean/UnsortedSegmentSum_1:output:0UnsortedSegmentMean/Maximum:z:0*
T0*
_output_shapes
:Q
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ы

GatherV2_2GatherV2concat:output:0Squeeze_3:output:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:Q
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : Щ

GatherV2_3GatherV2ones:output:0Squeeze_4:output:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:L
	Squeeze_7SqueezeGatherV2_3:output:0*
T0*
_output_shapes
:L
	Squeeze_8SqueezeGatherV2_2:output:0*
T0*
_output_shapes
:O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :А
concat_2ConcatV2Squeeze_8:output:0Squeeze_7:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:Б
EnsureShape_1EnsureShapeconcat_2:output:0*
T0*(
_output_shapes
:         А*
shape:         АФ
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallEnsureShape_1:output:0sequential_1_4846588sequential_1_4846590*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_4845541t
UnsortedSegmentMean_1/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_1/onesFill$UnsortedSegmentMean_1/Shape:output:0)UnsortedSegmentMean_1/ones/Const:output:0*
T0*
_output_shapes
:├
(UnsortedSegmentMean_1/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_1/ones:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
#UnsortedSegmentMean_1/strided_sliceStridedSlicen_i2UnsortedSegmentMean_1/strided_slice/stack:output:04UnsortedSegmentMean_1/strided_slice/stack_1:output:04UnsortedSegmentMean_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_1/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_1/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_1/subSub#UnsortedSegmentMean_1/Rank:output:0%UnsortedSegmentMean_1/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_1/ones_1/packedPackUnsortedSegmentMean_1/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_1/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_1/ones_1Fill,UnsortedSegmentMean_1/ones_1/packed:output:0+UnsortedSegmentMean_1/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_1/concatConcatV2,UnsortedSegmentMean_1/strided_slice:output:0%UnsortedSegmentMean_1/ones_1:output:0*UnsortedSegmentMean_1/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_1/ReshapeReshape1UnsortedSegmentMean_1/UnsortedSegmentSum:output:0%UnsortedSegmentMean_1/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_1/MaximumMaximum&UnsortedSegmentMean_1/Reshape:output:0(UnsortedSegmentMean_1/Maximum/y:output:0*
T0*
_output_shapes
:╧
*UnsortedSegmentMean_1/UnsortedSegmentSum_1UnsortedSegmentSum-sequential_1/StatefulPartitionedCall:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_1/truedivRealDiv3UnsortedSegmentMean_1/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_1/Maximum:z:0*
T0*
_output_shapes
:С
EnsureShape_2EnsureShape!UnsortedSegmentMean_1/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         А┬
!update_ip/StatefulPartitionedCallStatefulPartitionedCallEnsureShape_2:output:0ones:output:0update_ip_4846654update_ip_4846656update_ip_4846658*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_update_ip_layer_call_and_return_conditional_losses_4846653П
EnsureShape_3EnsureShapeUnsortedSegmentMean/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         Аь
)update_connection/StatefulPartitionedCallStatefulPartitionedCallEnsureShape_3:output:0concat:output:0update_connection_4846702update_connection_4846704update_connection_4846706*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_update_connection_layer_call_and_return_conditional_losses_4846701Q
GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B : ╢

GatherV2_4GatherV2*update_ip/StatefulPartitionedCall:output:0Squeeze_1:output:0GatherV2_4/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:Q
GatherV2_5/axisConst*
_output_shapes
: *
dtype0*
value	B : ╛

GatherV2_5GatherV22update_connection/StatefulPartitionedCall:output:0Squeeze_2:output:0GatherV2_5/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:L
	Squeeze_9SqueezeGatherV2_5:output:0*
T0*
_output_shapes
:M

Squeeze_10SqueezeGatherV2_4:output:0*
T0*
_output_shapes
:O
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B :Б
concat_3ConcatV2Squeeze_10:output:0Squeeze_9:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:Б
EnsureShape_4EnsureShapeconcat_3:output:0*
T0*(
_output_shapes
:         А*
shape:         АО
$sequential/StatefulPartitionedCall_1StatefulPartitionedCallEnsureShape_4:output:0sequential_4846553sequential_4846555*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4845456t
UnsortedSegmentMean_2/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_2/onesFill$UnsortedSegmentMean_2/Shape:output:0)UnsortedSegmentMean_2/ones/Const:output:0*
T0*
_output_shapes
:├
(UnsortedSegmentMean_2/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_2/ones:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
#UnsortedSegmentMean_2/strided_sliceStridedSlicen_c2UnsortedSegmentMean_2/strided_slice/stack:output:04UnsortedSegmentMean_2/strided_slice/stack_1:output:04UnsortedSegmentMean_2/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_2/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_2/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_2/subSub#UnsortedSegmentMean_2/Rank:output:0%UnsortedSegmentMean_2/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_2/ones_1/packedPackUnsortedSegmentMean_2/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_2/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_2/ones_1Fill,UnsortedSegmentMean_2/ones_1/packed:output:0+UnsortedSegmentMean_2/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_2/concatConcatV2,UnsortedSegmentMean_2/strided_slice:output:0%UnsortedSegmentMean_2/ones_1:output:0*UnsortedSegmentMean_2/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_2/ReshapeReshape1UnsortedSegmentMean_2/UnsortedSegmentSum:output:0%UnsortedSegmentMean_2/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_2/MaximumMaximum&UnsortedSegmentMean_2/Reshape:output:0(UnsortedSegmentMean_2/Maximum/y:output:0*
T0*
_output_shapes
:╧
*UnsortedSegmentMean_2/UnsortedSegmentSum_1UnsortedSegmentSum-sequential/StatefulPartitionedCall_1:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_2/truedivRealDiv3UnsortedSegmentMean_2/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_2/Maximum:z:0*
T0*
_output_shapes
:Q
GatherV2_6/axisConst*
_output_shapes
: *
dtype0*
value	B : ╛

GatherV2_6GatherV22update_connection/StatefulPartitionedCall:output:0Squeeze_3:output:0GatherV2_6/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:Q
GatherV2_7/axisConst*
_output_shapes
: *
dtype0*
value	B : ╢

GatherV2_7GatherV2*update_ip/StatefulPartitionedCall:output:0Squeeze_4:output:0GatherV2_7/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:M

Squeeze_11SqueezeGatherV2_7:output:0*
T0*
_output_shapes
:M

Squeeze_12SqueezeGatherV2_6:output:0*
T0*
_output_shapes
:O
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concat_4ConcatV2Squeeze_12:output:0Squeeze_11:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:Б
EnsureShape_5EnsureShapeconcat_4:output:0*
T0*(
_output_shapes
:         А*
shape:         АЦ
&sequential_1/StatefulPartitionedCall_1StatefulPartitionedCallEnsureShape_5:output:0sequential_1_4846588sequential_1_4846590*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_4845541t
UnsortedSegmentMean_3/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_3/onesFill$UnsortedSegmentMean_3/Shape:output:0)UnsortedSegmentMean_3/ones/Const:output:0*
T0*
_output_shapes
:├
(UnsortedSegmentMean_3/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_3/ones:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
#UnsortedSegmentMean_3/strided_sliceStridedSlicen_i2UnsortedSegmentMean_3/strided_slice/stack:output:04UnsortedSegmentMean_3/strided_slice/stack_1:output:04UnsortedSegmentMean_3/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_3/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_3/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_3/subSub#UnsortedSegmentMean_3/Rank:output:0%UnsortedSegmentMean_3/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_3/ones_1/packedPackUnsortedSegmentMean_3/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_3/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_3/ones_1Fill,UnsortedSegmentMean_3/ones_1/packed:output:0+UnsortedSegmentMean_3/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_3/concatConcatV2,UnsortedSegmentMean_3/strided_slice:output:0%UnsortedSegmentMean_3/ones_1:output:0*UnsortedSegmentMean_3/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_3/ReshapeReshape1UnsortedSegmentMean_3/UnsortedSegmentSum:output:0%UnsortedSegmentMean_3/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_3/MaximumMaximum&UnsortedSegmentMean_3/Reshape:output:0(UnsortedSegmentMean_3/Maximum/y:output:0*
T0*
_output_shapes
:╤
*UnsortedSegmentMean_3/UnsortedSegmentSum_1UnsortedSegmentSum/sequential_1/StatefulPartitionedCall_1:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_3/truedivRealDiv3UnsortedSegmentMean_3/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_3/Maximum:z:0*
T0*
_output_shapes
:С
EnsureShape_6EnsureShape!UnsortedSegmentMean_3/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         Ас
#update_ip/StatefulPartitionedCall_1StatefulPartitionedCallEnsureShape_6:output:0*update_ip/StatefulPartitionedCall:output:0update_ip_4846654update_ip_4846656update_ip_4846658*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_update_ip_layer_call_and_return_conditional_losses_4846653С
EnsureShape_7EnsureShape!UnsortedSegmentMean_2/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АС
+update_connection/StatefulPartitionedCall_1StatefulPartitionedCallEnsureShape_7:output:02update_connection/StatefulPartitionedCall:output:0update_connection_4846702update_connection_4846704update_connection_4846706*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_update_connection_layer_call_and_return_conditional_losses_4846701Q
GatherV2_8/axisConst*
_output_shapes
: *
dtype0*
value	B : ╕

GatherV2_8GatherV2,update_ip/StatefulPartitionedCall_1:output:0Squeeze_1:output:0GatherV2_8/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:Q
GatherV2_9/axisConst*
_output_shapes
: *
dtype0*
value	B : └

GatherV2_9GatherV24update_connection/StatefulPartitionedCall_1:output:0Squeeze_2:output:0GatherV2_9/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:M

Squeeze_13SqueezeGatherV2_9:output:0*
T0*
_output_shapes
:M

Squeeze_14SqueezeGatherV2_8:output:0*
T0*
_output_shapes
:O
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concat_5ConcatV2Squeeze_14:output:0Squeeze_13:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes
:Б
EnsureShape_8EnsureShapeconcat_5:output:0*
T0*(
_output_shapes
:         А*
shape:         АО
$sequential/StatefulPartitionedCall_2StatefulPartitionedCallEnsureShape_8:output:0sequential_4846553sequential_4846555*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4845456t
UnsortedSegmentMean_4/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_4/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_4/onesFill$UnsortedSegmentMean_4/Shape:output:0)UnsortedSegmentMean_4/ones/Const:output:0*
T0*
_output_shapes
:├
(UnsortedSegmentMean_4/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_4/ones:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
#UnsortedSegmentMean_4/strided_sliceStridedSlicen_c2UnsortedSegmentMean_4/strided_slice/stack:output:04UnsortedSegmentMean_4/strided_slice/stack_1:output:04UnsortedSegmentMean_4/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_4/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_4/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_4/subSub#UnsortedSegmentMean_4/Rank:output:0%UnsortedSegmentMean_4/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_4/ones_1/packedPackUnsortedSegmentMean_4/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_4/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_4/ones_1Fill,UnsortedSegmentMean_4/ones_1/packed:output:0+UnsortedSegmentMean_4/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_4/concatConcatV2,UnsortedSegmentMean_4/strided_slice:output:0%UnsortedSegmentMean_4/ones_1:output:0*UnsortedSegmentMean_4/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_4/ReshapeReshape1UnsortedSegmentMean_4/UnsortedSegmentSum:output:0%UnsortedSegmentMean_4/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_4/MaximumMaximum&UnsortedSegmentMean_4/Reshape:output:0(UnsortedSegmentMean_4/Maximum/y:output:0*
T0*
_output_shapes
:╧
*UnsortedSegmentMean_4/UnsortedSegmentSum_1UnsortedSegmentSum-sequential/StatefulPartitionedCall_2:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_4/truedivRealDiv3UnsortedSegmentMean_4/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_4/Maximum:z:0*
T0*
_output_shapes
:R
GatherV2_10/axisConst*
_output_shapes
: *
dtype0*
value	B : ┬
GatherV2_10GatherV24update_connection/StatefulPartitionedCall_1:output:0Squeeze_3:output:0GatherV2_10/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_11/axisConst*
_output_shapes
: *
dtype0*
value	B : ║
GatherV2_11GatherV2,update_ip/StatefulPartitionedCall_1:output:0Squeeze_4:output:0GatherV2_11/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_15SqueezeGatherV2_11:output:0*
T0*
_output_shapes
:N

Squeeze_16SqueezeGatherV2_10:output:0*
T0*
_output_shapes
:O
concat_6/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concat_6ConcatV2Squeeze_16:output:0Squeeze_15:output:0concat_6/axis:output:0*
N*
T0*
_output_shapes
:Б
EnsureShape_9EnsureShapeconcat_6:output:0*
T0*(
_output_shapes
:         А*
shape:         АЦ
&sequential_1/StatefulPartitionedCall_2StatefulPartitionedCallEnsureShape_9:output:0sequential_1_4846588sequential_1_4846590*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_4845541t
UnsortedSegmentMean_5/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_5/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_5/onesFill$UnsortedSegmentMean_5/Shape:output:0)UnsortedSegmentMean_5/ones/Const:output:0*
T0*
_output_shapes
:├
(UnsortedSegmentMean_5/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_5/ones:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
#UnsortedSegmentMean_5/strided_sliceStridedSlicen_i2UnsortedSegmentMean_5/strided_slice/stack:output:04UnsortedSegmentMean_5/strided_slice/stack_1:output:04UnsortedSegmentMean_5/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_5/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_5/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_5/subSub#UnsortedSegmentMean_5/Rank:output:0%UnsortedSegmentMean_5/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_5/ones_1/packedPackUnsortedSegmentMean_5/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_5/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_5/ones_1Fill,UnsortedSegmentMean_5/ones_1/packed:output:0+UnsortedSegmentMean_5/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_5/concatConcatV2,UnsortedSegmentMean_5/strided_slice:output:0%UnsortedSegmentMean_5/ones_1:output:0*UnsortedSegmentMean_5/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_5/ReshapeReshape1UnsortedSegmentMean_5/UnsortedSegmentSum:output:0%UnsortedSegmentMean_5/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_5/MaximumMaximum&UnsortedSegmentMean_5/Reshape:output:0(UnsortedSegmentMean_5/Maximum/y:output:0*
T0*
_output_shapes
:╤
*UnsortedSegmentMean_5/UnsortedSegmentSum_1UnsortedSegmentSum/sequential_1/StatefulPartitionedCall_2:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_5/truedivRealDiv3UnsortedSegmentMean_5/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_5/Maximum:z:0*
T0*
_output_shapes
:Т
EnsureShape_10EnsureShape!UnsortedSegmentMean_5/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         Аф
#update_ip/StatefulPartitionedCall_2StatefulPartitionedCallEnsureShape_10:output:0,update_ip/StatefulPartitionedCall_1:output:0update_ip_4846654update_ip_4846656update_ip_4846658*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_update_ip_layer_call_and_return_conditional_losses_4846653Т
EnsureShape_11EnsureShape!UnsortedSegmentMean_4/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АФ
+update_connection/StatefulPartitionedCall_2StatefulPartitionedCallEnsureShape_11:output:04update_connection/StatefulPartitionedCall_1:output:0update_connection_4846702update_connection_4846704update_connection_4846706*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_update_connection_layer_call_and_return_conditional_losses_4846701R
GatherV2_12/axisConst*
_output_shapes
: *
dtype0*
value	B : ║
GatherV2_12GatherV2,update_ip/StatefulPartitionedCall_2:output:0Squeeze_1:output:0GatherV2_12/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_13/axisConst*
_output_shapes
: *
dtype0*
value	B : ┬
GatherV2_13GatherV24update_connection/StatefulPartitionedCall_2:output:0Squeeze_2:output:0GatherV2_13/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_17SqueezeGatherV2_13:output:0*
T0*
_output_shapes
:N

Squeeze_18SqueezeGatherV2_12:output:0*
T0*
_output_shapes
:O
concat_7/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concat_7ConcatV2Squeeze_18:output:0Squeeze_17:output:0concat_7/axis:output:0*
N*
T0*
_output_shapes
:В
EnsureShape_12EnsureShapeconcat_7:output:0*
T0*(
_output_shapes
:         А*
shape:         АП
$sequential/StatefulPartitionedCall_3StatefulPartitionedCallEnsureShape_12:output:0sequential_4846553sequential_4846555*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4845456t
UnsortedSegmentMean_6/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_6/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_6/onesFill$UnsortedSegmentMean_6/Shape:output:0)UnsortedSegmentMean_6/ones/Const:output:0*
T0*
_output_shapes
:├
(UnsortedSegmentMean_6/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_6/ones:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
#UnsortedSegmentMean_6/strided_sliceStridedSlicen_c2UnsortedSegmentMean_6/strided_slice/stack:output:04UnsortedSegmentMean_6/strided_slice/stack_1:output:04UnsortedSegmentMean_6/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_6/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_6/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_6/subSub#UnsortedSegmentMean_6/Rank:output:0%UnsortedSegmentMean_6/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_6/ones_1/packedPackUnsortedSegmentMean_6/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_6/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_6/ones_1Fill,UnsortedSegmentMean_6/ones_1/packed:output:0+UnsortedSegmentMean_6/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_6/concatConcatV2,UnsortedSegmentMean_6/strided_slice:output:0%UnsortedSegmentMean_6/ones_1:output:0*UnsortedSegmentMean_6/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_6/ReshapeReshape1UnsortedSegmentMean_6/UnsortedSegmentSum:output:0%UnsortedSegmentMean_6/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_6/MaximumMaximum&UnsortedSegmentMean_6/Reshape:output:0(UnsortedSegmentMean_6/Maximum/y:output:0*
T0*
_output_shapes
:╧
*UnsortedSegmentMean_6/UnsortedSegmentSum_1UnsortedSegmentSum-sequential/StatefulPartitionedCall_3:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_6/truedivRealDiv3UnsortedSegmentMean_6/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_6/Maximum:z:0*
T0*
_output_shapes
:R
GatherV2_14/axisConst*
_output_shapes
: *
dtype0*
value	B : ┬
GatherV2_14GatherV24update_connection/StatefulPartitionedCall_2:output:0Squeeze_3:output:0GatherV2_14/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_15/axisConst*
_output_shapes
: *
dtype0*
value	B : ║
GatherV2_15GatherV2,update_ip/StatefulPartitionedCall_2:output:0Squeeze_4:output:0GatherV2_15/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_19SqueezeGatherV2_15:output:0*
T0*
_output_shapes
:N

Squeeze_20SqueezeGatherV2_14:output:0*
T0*
_output_shapes
:O
concat_8/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concat_8ConcatV2Squeeze_20:output:0Squeeze_19:output:0concat_8/axis:output:0*
N*
T0*
_output_shapes
:В
EnsureShape_13EnsureShapeconcat_8:output:0*
T0*(
_output_shapes
:         А*
shape:         АЧ
&sequential_1/StatefulPartitionedCall_3StatefulPartitionedCallEnsureShape_13:output:0sequential_1_4846588sequential_1_4846590*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_4845541t
UnsortedSegmentMean_7/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_7/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_7/onesFill$UnsortedSegmentMean_7/Shape:output:0)UnsortedSegmentMean_7/ones/Const:output:0*
T0*
_output_shapes
:├
(UnsortedSegmentMean_7/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_7/ones:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
#UnsortedSegmentMean_7/strided_sliceStridedSlicen_i2UnsortedSegmentMean_7/strided_slice/stack:output:04UnsortedSegmentMean_7/strided_slice/stack_1:output:04UnsortedSegmentMean_7/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_7/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_7/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_7/subSub#UnsortedSegmentMean_7/Rank:output:0%UnsortedSegmentMean_7/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_7/ones_1/packedPackUnsortedSegmentMean_7/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_7/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_7/ones_1Fill,UnsortedSegmentMean_7/ones_1/packed:output:0+UnsortedSegmentMean_7/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_7/concatConcatV2,UnsortedSegmentMean_7/strided_slice:output:0%UnsortedSegmentMean_7/ones_1:output:0*UnsortedSegmentMean_7/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_7/ReshapeReshape1UnsortedSegmentMean_7/UnsortedSegmentSum:output:0%UnsortedSegmentMean_7/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_7/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_7/MaximumMaximum&UnsortedSegmentMean_7/Reshape:output:0(UnsortedSegmentMean_7/Maximum/y:output:0*
T0*
_output_shapes
:╤
*UnsortedSegmentMean_7/UnsortedSegmentSum_1UnsortedSegmentSum/sequential_1/StatefulPartitionedCall_3:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_7/truedivRealDiv3UnsortedSegmentMean_7/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_7/Maximum:z:0*
T0*
_output_shapes
:Т
EnsureShape_14EnsureShape!UnsortedSegmentMean_7/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         Аф
#update_ip/StatefulPartitionedCall_3StatefulPartitionedCallEnsureShape_14:output:0,update_ip/StatefulPartitionedCall_2:output:0update_ip_4846654update_ip_4846656update_ip_4846658*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_update_ip_layer_call_and_return_conditional_losses_4846653Т
EnsureShape_15EnsureShape!UnsortedSegmentMean_6/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АФ
+update_connection/StatefulPartitionedCall_3StatefulPartitionedCallEnsureShape_15:output:04update_connection/StatefulPartitionedCall_2:output:0update_connection_4846702update_connection_4846704update_connection_4846706*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_update_connection_layer_call_and_return_conditional_losses_4846701R
GatherV2_16/axisConst*
_output_shapes
: *
dtype0*
value	B : ║
GatherV2_16GatherV2,update_ip/StatefulPartitionedCall_3:output:0Squeeze_1:output:0GatherV2_16/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_17/axisConst*
_output_shapes
: *
dtype0*
value	B : ┬
GatherV2_17GatherV24update_connection/StatefulPartitionedCall_3:output:0Squeeze_2:output:0GatherV2_17/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_21SqueezeGatherV2_17:output:0*
T0*
_output_shapes
:N

Squeeze_22SqueezeGatherV2_16:output:0*
T0*
_output_shapes
:O
concat_9/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concat_9ConcatV2Squeeze_22:output:0Squeeze_21:output:0concat_9/axis:output:0*
N*
T0*
_output_shapes
:В
EnsureShape_16EnsureShapeconcat_9:output:0*
T0*(
_output_shapes
:         А*
shape:         АП
$sequential/StatefulPartitionedCall_4StatefulPartitionedCallEnsureShape_16:output:0sequential_4846553sequential_4846555*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4845456t
UnsortedSegmentMean_8/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_8/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_8/onesFill$UnsortedSegmentMean_8/Shape:output:0)UnsortedSegmentMean_8/ones/Const:output:0*
T0*
_output_shapes
:├
(UnsortedSegmentMean_8/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_8/ones:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
#UnsortedSegmentMean_8/strided_sliceStridedSlicen_c2UnsortedSegmentMean_8/strided_slice/stack:output:04UnsortedSegmentMean_8/strided_slice/stack_1:output:04UnsortedSegmentMean_8/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_8/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_8/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_8/subSub#UnsortedSegmentMean_8/Rank:output:0%UnsortedSegmentMean_8/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_8/ones_1/packedPackUnsortedSegmentMean_8/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_8/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_8/ones_1Fill,UnsortedSegmentMean_8/ones_1/packed:output:0+UnsortedSegmentMean_8/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_8/concatConcatV2,UnsortedSegmentMean_8/strided_slice:output:0%UnsortedSegmentMean_8/ones_1:output:0*UnsortedSegmentMean_8/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_8/ReshapeReshape1UnsortedSegmentMean_8/UnsortedSegmentSum:output:0%UnsortedSegmentMean_8/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_8/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_8/MaximumMaximum&UnsortedSegmentMean_8/Reshape:output:0(UnsortedSegmentMean_8/Maximum/y:output:0*
T0*
_output_shapes
:╧
*UnsortedSegmentMean_8/UnsortedSegmentSum_1UnsortedSegmentSum-sequential/StatefulPartitionedCall_4:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_8/truedivRealDiv3UnsortedSegmentMean_8/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_8/Maximum:z:0*
T0*
_output_shapes
:R
GatherV2_18/axisConst*
_output_shapes
: *
dtype0*
value	B : ┬
GatherV2_18GatherV24update_connection/StatefulPartitionedCall_3:output:0Squeeze_3:output:0GatherV2_18/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_19/axisConst*
_output_shapes
: *
dtype0*
value	B : ║
GatherV2_19GatherV2,update_ip/StatefulPartitionedCall_3:output:0Squeeze_4:output:0GatherV2_19/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_23SqueezeGatherV2_19:output:0*
T0*
_output_shapes
:N

Squeeze_24SqueezeGatherV2_18:output:0*
T0*
_output_shapes
:P
concat_10/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_10ConcatV2Squeeze_24:output:0Squeeze_23:output:0concat_10/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_17EnsureShapeconcat_10:output:0*
T0*(
_output_shapes
:         А*
shape:         АЧ
&sequential_1/StatefulPartitionedCall_4StatefulPartitionedCallEnsureShape_17:output:0sequential_1_4846588sequential_1_4846590*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_4845541t
UnsortedSegmentMean_9/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_9/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_9/onesFill$UnsortedSegmentMean_9/Shape:output:0)UnsortedSegmentMean_9/ones/Const:output:0*
T0*
_output_shapes
:├
(UnsortedSegmentMean_9/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_9/ones:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
#UnsortedSegmentMean_9/strided_sliceStridedSlicen_i2UnsortedSegmentMean_9/strided_slice/stack:output:04UnsortedSegmentMean_9/strided_slice/stack_1:output:04UnsortedSegmentMean_9/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_9/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_9/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_9/subSub#UnsortedSegmentMean_9/Rank:output:0%UnsortedSegmentMean_9/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_9/ones_1/packedPackUnsortedSegmentMean_9/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_9/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_9/ones_1Fill,UnsortedSegmentMean_9/ones_1/packed:output:0+UnsortedSegmentMean_9/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_9/concatConcatV2,UnsortedSegmentMean_9/strided_slice:output:0%UnsortedSegmentMean_9/ones_1:output:0*UnsortedSegmentMean_9/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_9/ReshapeReshape1UnsortedSegmentMean_9/UnsortedSegmentSum:output:0%UnsortedSegmentMean_9/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_9/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_9/MaximumMaximum&UnsortedSegmentMean_9/Reshape:output:0(UnsortedSegmentMean_9/Maximum/y:output:0*
T0*
_output_shapes
:╤
*UnsortedSegmentMean_9/UnsortedSegmentSum_1UnsortedSegmentSum/sequential_1/StatefulPartitionedCall_4:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_9/truedivRealDiv3UnsortedSegmentMean_9/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_9/Maximum:z:0*
T0*
_output_shapes
:Т
EnsureShape_18EnsureShape!UnsortedSegmentMean_9/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         Аф
#update_ip/StatefulPartitionedCall_4StatefulPartitionedCallEnsureShape_18:output:0,update_ip/StatefulPartitionedCall_3:output:0update_ip_4846654update_ip_4846656update_ip_4846658*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_update_ip_layer_call_and_return_conditional_losses_4846653Т
EnsureShape_19EnsureShape!UnsortedSegmentMean_8/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АФ
+update_connection/StatefulPartitionedCall_4StatefulPartitionedCallEnsureShape_19:output:04update_connection/StatefulPartitionedCall_3:output:0update_connection_4846702update_connection_4846704update_connection_4846706*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_update_connection_layer_call_and_return_conditional_losses_4846701R
GatherV2_20/axisConst*
_output_shapes
: *
dtype0*
value	B : ║
GatherV2_20GatherV2,update_ip/StatefulPartitionedCall_4:output:0Squeeze_1:output:0GatherV2_20/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_21/axisConst*
_output_shapes
: *
dtype0*
value	B : ┬
GatherV2_21GatherV24update_connection/StatefulPartitionedCall_4:output:0Squeeze_2:output:0GatherV2_21/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_25SqueezeGatherV2_21:output:0*
T0*
_output_shapes
:N

Squeeze_26SqueezeGatherV2_20:output:0*
T0*
_output_shapes
:P
concat_11/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_11ConcatV2Squeeze_26:output:0Squeeze_25:output:0concat_11/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_20EnsureShapeconcat_11:output:0*
T0*(
_output_shapes
:         А*
shape:         АП
$sequential/StatefulPartitionedCall_5StatefulPartitionedCallEnsureShape_20:output:0sequential_4846553sequential_4846555*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4845456u
UnsortedSegmentMean_10/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧f
!UnsortedSegmentMean_10/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
UnsortedSegmentMean_10/onesFill%UnsortedSegmentMean_10/Shape:output:0*UnsortedSegmentMean_10/ones/Const:output:0*
T0*
_output_shapes
:┼
)UnsortedSegmentMean_10/UnsortedSegmentSumUnsortedSegmentSum$UnsortedSegmentMean_10/ones:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:t
*UnsortedSegmentMean_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
$UnsortedSegmentMean_10/strided_sliceStridedSlicen_c3UnsortedSegmentMean_10/strided_slice/stack:output:05UnsortedSegmentMean_10/strided_slice/stack_1:output:05UnsortedSegmentMean_10/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask]
UnsortedSegmentMean_10/RankConst*
_output_shapes
: *
dtype0*
value	B :Z
UnsortedSegmentMean_10/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Р
UnsortedSegmentMean_10/subSub$UnsortedSegmentMean_10/Rank:output:0&UnsortedSegmentMean_10/Rank_1:output:0*
T0*
_output_shapes
: z
$UnsortedSegmentMean_10/ones_1/packedPackUnsortedSegmentMean_10/sub:z:0*
N*
T0*
_output_shapes
:e
#UnsortedSegmentMean_10/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R░
UnsortedSegmentMean_10/ones_1Fill-UnsortedSegmentMean_10/ones_1/packed:output:0,UnsortedSegmentMean_10/ones_1/Const:output:0*
T0	*#
_output_shapes
:         d
"UnsortedSegmentMean_10/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
UnsortedSegmentMean_10/concatConcatV2-UnsortedSegmentMean_10/strided_slice:output:0&UnsortedSegmentMean_10/ones_1:output:0+UnsortedSegmentMean_10/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         ╢
UnsortedSegmentMean_10/ReshapeReshape2UnsortedSegmentMean_10/UnsortedSegmentSum:output:0&UnsortedSegmentMean_10/concat:output:0*
Tshape0	*
T0*
_output_shapes
:e
 UnsortedSegmentMean_10/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
UnsortedSegmentMean_10/MaximumMaximum'UnsortedSegmentMean_10/Reshape:output:0)UnsortedSegmentMean_10/Maximum/y:output:0*
T0*
_output_shapes
:╨
+UnsortedSegmentMean_10/UnsortedSegmentSum_1UnsortedSegmentSum-sequential/StatefulPartitionedCall_5:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:ж
UnsortedSegmentMean_10/truedivRealDiv4UnsortedSegmentMean_10/UnsortedSegmentSum_1:output:0"UnsortedSegmentMean_10/Maximum:z:0*
T0*
_output_shapes
:R
GatherV2_22/axisConst*
_output_shapes
: *
dtype0*
value	B : ┬
GatherV2_22GatherV24update_connection/StatefulPartitionedCall_4:output:0Squeeze_3:output:0GatherV2_22/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_23/axisConst*
_output_shapes
: *
dtype0*
value	B : ║
GatherV2_23GatherV2,update_ip/StatefulPartitionedCall_4:output:0Squeeze_4:output:0GatherV2_23/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_27SqueezeGatherV2_23:output:0*
T0*
_output_shapes
:N

Squeeze_28SqueezeGatherV2_22:output:0*
T0*
_output_shapes
:P
concat_12/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_12ConcatV2Squeeze_28:output:0Squeeze_27:output:0concat_12/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_21EnsureShapeconcat_12:output:0*
T0*(
_output_shapes
:         А*
shape:         АЧ
&sequential_1/StatefulPartitionedCall_5StatefulPartitionedCallEnsureShape_21:output:0sequential_1_4846588sequential_1_4846590*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_4845541u
UnsortedSegmentMean_11/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧f
!UnsortedSegmentMean_11/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
UnsortedSegmentMean_11/onesFill%UnsortedSegmentMean_11/Shape:output:0*UnsortedSegmentMean_11/ones/Const:output:0*
T0*
_output_shapes
:┼
)UnsortedSegmentMean_11/UnsortedSegmentSumUnsortedSegmentSum$UnsortedSegmentMean_11/ones:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:t
*UnsortedSegmentMean_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
$UnsortedSegmentMean_11/strided_sliceStridedSlicen_i3UnsortedSegmentMean_11/strided_slice/stack:output:05UnsortedSegmentMean_11/strided_slice/stack_1:output:05UnsortedSegmentMean_11/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask]
UnsortedSegmentMean_11/RankConst*
_output_shapes
: *
dtype0*
value	B :Z
UnsortedSegmentMean_11/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Р
UnsortedSegmentMean_11/subSub$UnsortedSegmentMean_11/Rank:output:0&UnsortedSegmentMean_11/Rank_1:output:0*
T0*
_output_shapes
: z
$UnsortedSegmentMean_11/ones_1/packedPackUnsortedSegmentMean_11/sub:z:0*
N*
T0*
_output_shapes
:e
#UnsortedSegmentMean_11/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R░
UnsortedSegmentMean_11/ones_1Fill-UnsortedSegmentMean_11/ones_1/packed:output:0,UnsortedSegmentMean_11/ones_1/Const:output:0*
T0	*#
_output_shapes
:         d
"UnsortedSegmentMean_11/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
UnsortedSegmentMean_11/concatConcatV2-UnsortedSegmentMean_11/strided_slice:output:0&UnsortedSegmentMean_11/ones_1:output:0+UnsortedSegmentMean_11/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         ╢
UnsortedSegmentMean_11/ReshapeReshape2UnsortedSegmentMean_11/UnsortedSegmentSum:output:0&UnsortedSegmentMean_11/concat:output:0*
Tshape0	*
T0*
_output_shapes
:e
 UnsortedSegmentMean_11/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
UnsortedSegmentMean_11/MaximumMaximum'UnsortedSegmentMean_11/Reshape:output:0)UnsortedSegmentMean_11/Maximum/y:output:0*
T0*
_output_shapes
:╥
+UnsortedSegmentMean_11/UnsortedSegmentSum_1UnsortedSegmentSum/sequential_1/StatefulPartitionedCall_5:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:ж
UnsortedSegmentMean_11/truedivRealDiv4UnsortedSegmentMean_11/UnsortedSegmentSum_1:output:0"UnsortedSegmentMean_11/Maximum:z:0*
T0*
_output_shapes
:У
EnsureShape_22EnsureShape"UnsortedSegmentMean_11/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         Аф
#update_ip/StatefulPartitionedCall_5StatefulPartitionedCallEnsureShape_22:output:0,update_ip/StatefulPartitionedCall_4:output:0update_ip_4846654update_ip_4846656update_ip_4846658*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_update_ip_layer_call_and_return_conditional_losses_4846653У
EnsureShape_23EnsureShape"UnsortedSegmentMean_10/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АФ
+update_connection/StatefulPartitionedCall_5StatefulPartitionedCallEnsureShape_23:output:04update_connection/StatefulPartitionedCall_4:output:0update_connection_4846702update_connection_4846704update_connection_4846706*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_update_connection_layer_call_and_return_conditional_losses_4846701R
GatherV2_24/axisConst*
_output_shapes
: *
dtype0*
value	B : ║
GatherV2_24GatherV2,update_ip/StatefulPartitionedCall_5:output:0Squeeze_1:output:0GatherV2_24/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_25/axisConst*
_output_shapes
: *
dtype0*
value	B : ┬
GatherV2_25GatherV24update_connection/StatefulPartitionedCall_5:output:0Squeeze_2:output:0GatherV2_25/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_29SqueezeGatherV2_25:output:0*
T0*
_output_shapes
:N

Squeeze_30SqueezeGatherV2_24:output:0*
T0*
_output_shapes
:P
concat_13/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_13ConcatV2Squeeze_30:output:0Squeeze_29:output:0concat_13/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_24EnsureShapeconcat_13:output:0*
T0*(
_output_shapes
:         А*
shape:         АП
$sequential/StatefulPartitionedCall_6StatefulPartitionedCallEnsureShape_24:output:0sequential_4846553sequential_4846555*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4845456u
UnsortedSegmentMean_12/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧f
!UnsortedSegmentMean_12/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
UnsortedSegmentMean_12/onesFill%UnsortedSegmentMean_12/Shape:output:0*UnsortedSegmentMean_12/ones/Const:output:0*
T0*
_output_shapes
:┼
)UnsortedSegmentMean_12/UnsortedSegmentSumUnsortedSegmentSum$UnsortedSegmentMean_12/ones:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:t
*UnsortedSegmentMean_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
$UnsortedSegmentMean_12/strided_sliceStridedSlicen_c3UnsortedSegmentMean_12/strided_slice/stack:output:05UnsortedSegmentMean_12/strided_slice/stack_1:output:05UnsortedSegmentMean_12/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask]
UnsortedSegmentMean_12/RankConst*
_output_shapes
: *
dtype0*
value	B :Z
UnsortedSegmentMean_12/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Р
UnsortedSegmentMean_12/subSub$UnsortedSegmentMean_12/Rank:output:0&UnsortedSegmentMean_12/Rank_1:output:0*
T0*
_output_shapes
: z
$UnsortedSegmentMean_12/ones_1/packedPackUnsortedSegmentMean_12/sub:z:0*
N*
T0*
_output_shapes
:e
#UnsortedSegmentMean_12/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R░
UnsortedSegmentMean_12/ones_1Fill-UnsortedSegmentMean_12/ones_1/packed:output:0,UnsortedSegmentMean_12/ones_1/Const:output:0*
T0	*#
_output_shapes
:         d
"UnsortedSegmentMean_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
UnsortedSegmentMean_12/concatConcatV2-UnsortedSegmentMean_12/strided_slice:output:0&UnsortedSegmentMean_12/ones_1:output:0+UnsortedSegmentMean_12/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         ╢
UnsortedSegmentMean_12/ReshapeReshape2UnsortedSegmentMean_12/UnsortedSegmentSum:output:0&UnsortedSegmentMean_12/concat:output:0*
Tshape0	*
T0*
_output_shapes
:e
 UnsortedSegmentMean_12/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
UnsortedSegmentMean_12/MaximumMaximum'UnsortedSegmentMean_12/Reshape:output:0)UnsortedSegmentMean_12/Maximum/y:output:0*
T0*
_output_shapes
:╨
+UnsortedSegmentMean_12/UnsortedSegmentSum_1UnsortedSegmentSum-sequential/StatefulPartitionedCall_6:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:ж
UnsortedSegmentMean_12/truedivRealDiv4UnsortedSegmentMean_12/UnsortedSegmentSum_1:output:0"UnsortedSegmentMean_12/Maximum:z:0*
T0*
_output_shapes
:R
GatherV2_26/axisConst*
_output_shapes
: *
dtype0*
value	B : ┬
GatherV2_26GatherV24update_connection/StatefulPartitionedCall_5:output:0Squeeze_3:output:0GatherV2_26/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_27/axisConst*
_output_shapes
: *
dtype0*
value	B : ║
GatherV2_27GatherV2,update_ip/StatefulPartitionedCall_5:output:0Squeeze_4:output:0GatherV2_27/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_31SqueezeGatherV2_27:output:0*
T0*
_output_shapes
:N

Squeeze_32SqueezeGatherV2_26:output:0*
T0*
_output_shapes
:P
concat_14/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_14ConcatV2Squeeze_32:output:0Squeeze_31:output:0concat_14/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_25EnsureShapeconcat_14:output:0*
T0*(
_output_shapes
:         А*
shape:         АЧ
&sequential_1/StatefulPartitionedCall_6StatefulPartitionedCallEnsureShape_25:output:0sequential_1_4846588sequential_1_4846590*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_4845541u
UnsortedSegmentMean_13/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧f
!UnsortedSegmentMean_13/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
UnsortedSegmentMean_13/onesFill%UnsortedSegmentMean_13/Shape:output:0*UnsortedSegmentMean_13/ones/Const:output:0*
T0*
_output_shapes
:┼
)UnsortedSegmentMean_13/UnsortedSegmentSumUnsortedSegmentSum$UnsortedSegmentMean_13/ones:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:t
*UnsortedSegmentMean_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
$UnsortedSegmentMean_13/strided_sliceStridedSlicen_i3UnsortedSegmentMean_13/strided_slice/stack:output:05UnsortedSegmentMean_13/strided_slice/stack_1:output:05UnsortedSegmentMean_13/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask]
UnsortedSegmentMean_13/RankConst*
_output_shapes
: *
dtype0*
value	B :Z
UnsortedSegmentMean_13/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Р
UnsortedSegmentMean_13/subSub$UnsortedSegmentMean_13/Rank:output:0&UnsortedSegmentMean_13/Rank_1:output:0*
T0*
_output_shapes
: z
$UnsortedSegmentMean_13/ones_1/packedPackUnsortedSegmentMean_13/sub:z:0*
N*
T0*
_output_shapes
:e
#UnsortedSegmentMean_13/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R░
UnsortedSegmentMean_13/ones_1Fill-UnsortedSegmentMean_13/ones_1/packed:output:0,UnsortedSegmentMean_13/ones_1/Const:output:0*
T0	*#
_output_shapes
:         d
"UnsortedSegmentMean_13/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
UnsortedSegmentMean_13/concatConcatV2-UnsortedSegmentMean_13/strided_slice:output:0&UnsortedSegmentMean_13/ones_1:output:0+UnsortedSegmentMean_13/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         ╢
UnsortedSegmentMean_13/ReshapeReshape2UnsortedSegmentMean_13/UnsortedSegmentSum:output:0&UnsortedSegmentMean_13/concat:output:0*
Tshape0	*
T0*
_output_shapes
:e
 UnsortedSegmentMean_13/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
UnsortedSegmentMean_13/MaximumMaximum'UnsortedSegmentMean_13/Reshape:output:0)UnsortedSegmentMean_13/Maximum/y:output:0*
T0*
_output_shapes
:╥
+UnsortedSegmentMean_13/UnsortedSegmentSum_1UnsortedSegmentSum/sequential_1/StatefulPartitionedCall_6:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:ж
UnsortedSegmentMean_13/truedivRealDiv4UnsortedSegmentMean_13/UnsortedSegmentSum_1:output:0"UnsortedSegmentMean_13/Maximum:z:0*
T0*
_output_shapes
:У
EnsureShape_26EnsureShape"UnsortedSegmentMean_13/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         Аф
#update_ip/StatefulPartitionedCall_6StatefulPartitionedCallEnsureShape_26:output:0,update_ip/StatefulPartitionedCall_5:output:0update_ip_4846654update_ip_4846656update_ip_4846658*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_update_ip_layer_call_and_return_conditional_losses_4846653У
EnsureShape_27EnsureShape"UnsortedSegmentMean_12/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АФ
+update_connection/StatefulPartitionedCall_6StatefulPartitionedCallEnsureShape_27:output:04update_connection/StatefulPartitionedCall_5:output:0update_connection_4846702update_connection_4846704update_connection_4846706*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_update_connection_layer_call_and_return_conditional_losses_4846701R
GatherV2_28/axisConst*
_output_shapes
: *
dtype0*
value	B : ║
GatherV2_28GatherV2,update_ip/StatefulPartitionedCall_6:output:0Squeeze_1:output:0GatherV2_28/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_29/axisConst*
_output_shapes
: *
dtype0*
value	B : ┬
GatherV2_29GatherV24update_connection/StatefulPartitionedCall_6:output:0Squeeze_2:output:0GatherV2_29/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_33SqueezeGatherV2_29:output:0*
T0*
_output_shapes
:N

Squeeze_34SqueezeGatherV2_28:output:0*
T0*
_output_shapes
:P
concat_15/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_15ConcatV2Squeeze_34:output:0Squeeze_33:output:0concat_15/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_28EnsureShapeconcat_15:output:0*
T0*(
_output_shapes
:         А*
shape:         АП
$sequential/StatefulPartitionedCall_7StatefulPartitionedCallEnsureShape_28:output:0sequential_4846553sequential_4846555*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4845456u
UnsortedSegmentMean_14/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧f
!UnsortedSegmentMean_14/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
UnsortedSegmentMean_14/onesFill%UnsortedSegmentMean_14/Shape:output:0*UnsortedSegmentMean_14/ones/Const:output:0*
T0*
_output_shapes
:┼
)UnsortedSegmentMean_14/UnsortedSegmentSumUnsortedSegmentSum$UnsortedSegmentMean_14/ones:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:t
*UnsortedSegmentMean_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
$UnsortedSegmentMean_14/strided_sliceStridedSlicen_c3UnsortedSegmentMean_14/strided_slice/stack:output:05UnsortedSegmentMean_14/strided_slice/stack_1:output:05UnsortedSegmentMean_14/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask]
UnsortedSegmentMean_14/RankConst*
_output_shapes
: *
dtype0*
value	B :Z
UnsortedSegmentMean_14/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Р
UnsortedSegmentMean_14/subSub$UnsortedSegmentMean_14/Rank:output:0&UnsortedSegmentMean_14/Rank_1:output:0*
T0*
_output_shapes
: z
$UnsortedSegmentMean_14/ones_1/packedPackUnsortedSegmentMean_14/sub:z:0*
N*
T0*
_output_shapes
:e
#UnsortedSegmentMean_14/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R░
UnsortedSegmentMean_14/ones_1Fill-UnsortedSegmentMean_14/ones_1/packed:output:0,UnsortedSegmentMean_14/ones_1/Const:output:0*
T0	*#
_output_shapes
:         d
"UnsortedSegmentMean_14/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
UnsortedSegmentMean_14/concatConcatV2-UnsortedSegmentMean_14/strided_slice:output:0&UnsortedSegmentMean_14/ones_1:output:0+UnsortedSegmentMean_14/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         ╢
UnsortedSegmentMean_14/ReshapeReshape2UnsortedSegmentMean_14/UnsortedSegmentSum:output:0&UnsortedSegmentMean_14/concat:output:0*
Tshape0	*
T0*
_output_shapes
:e
 UnsortedSegmentMean_14/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
UnsortedSegmentMean_14/MaximumMaximum'UnsortedSegmentMean_14/Reshape:output:0)UnsortedSegmentMean_14/Maximum/y:output:0*
T0*
_output_shapes
:╨
+UnsortedSegmentMean_14/UnsortedSegmentSum_1UnsortedSegmentSum-sequential/StatefulPartitionedCall_7:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:ж
UnsortedSegmentMean_14/truedivRealDiv4UnsortedSegmentMean_14/UnsortedSegmentSum_1:output:0"UnsortedSegmentMean_14/Maximum:z:0*
T0*
_output_shapes
:R
GatherV2_30/axisConst*
_output_shapes
: *
dtype0*
value	B : ┬
GatherV2_30GatherV24update_connection/StatefulPartitionedCall_6:output:0Squeeze_3:output:0GatherV2_30/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_31/axisConst*
_output_shapes
: *
dtype0*
value	B : ║
GatherV2_31GatherV2,update_ip/StatefulPartitionedCall_6:output:0Squeeze_4:output:0GatherV2_31/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_35SqueezeGatherV2_31:output:0*
T0*
_output_shapes
:N

Squeeze_36SqueezeGatherV2_30:output:0*
T0*
_output_shapes
:P
concat_16/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_16ConcatV2Squeeze_36:output:0Squeeze_35:output:0concat_16/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_29EnsureShapeconcat_16:output:0*
T0*(
_output_shapes
:         А*
shape:         АЧ
&sequential_1/StatefulPartitionedCall_7StatefulPartitionedCallEnsureShape_29:output:0sequential_1_4846588sequential_1_4846590*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_4845541u
UnsortedSegmentMean_15/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧f
!UnsortedSegmentMean_15/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
UnsortedSegmentMean_15/onesFill%UnsortedSegmentMean_15/Shape:output:0*UnsortedSegmentMean_15/ones/Const:output:0*
T0*
_output_shapes
:┼
)UnsortedSegmentMean_15/UnsortedSegmentSumUnsortedSegmentSum$UnsortedSegmentMean_15/ones:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:t
*UnsortedSegmentMean_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
$UnsortedSegmentMean_15/strided_sliceStridedSlicen_i3UnsortedSegmentMean_15/strided_slice/stack:output:05UnsortedSegmentMean_15/strided_slice/stack_1:output:05UnsortedSegmentMean_15/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask]
UnsortedSegmentMean_15/RankConst*
_output_shapes
: *
dtype0*
value	B :Z
UnsortedSegmentMean_15/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Р
UnsortedSegmentMean_15/subSub$UnsortedSegmentMean_15/Rank:output:0&UnsortedSegmentMean_15/Rank_1:output:0*
T0*
_output_shapes
: z
$UnsortedSegmentMean_15/ones_1/packedPackUnsortedSegmentMean_15/sub:z:0*
N*
T0*
_output_shapes
:e
#UnsortedSegmentMean_15/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R░
UnsortedSegmentMean_15/ones_1Fill-UnsortedSegmentMean_15/ones_1/packed:output:0,UnsortedSegmentMean_15/ones_1/Const:output:0*
T0	*#
_output_shapes
:         d
"UnsortedSegmentMean_15/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
UnsortedSegmentMean_15/concatConcatV2-UnsortedSegmentMean_15/strided_slice:output:0&UnsortedSegmentMean_15/ones_1:output:0+UnsortedSegmentMean_15/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         ╢
UnsortedSegmentMean_15/ReshapeReshape2UnsortedSegmentMean_15/UnsortedSegmentSum:output:0&UnsortedSegmentMean_15/concat:output:0*
Tshape0	*
T0*
_output_shapes
:e
 UnsortedSegmentMean_15/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
UnsortedSegmentMean_15/MaximumMaximum'UnsortedSegmentMean_15/Reshape:output:0)UnsortedSegmentMean_15/Maximum/y:output:0*
T0*
_output_shapes
:╥
+UnsortedSegmentMean_15/UnsortedSegmentSum_1UnsortedSegmentSum/sequential_1/StatefulPartitionedCall_7:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:ж
UnsortedSegmentMean_15/truedivRealDiv4UnsortedSegmentMean_15/UnsortedSegmentSum_1:output:0"UnsortedSegmentMean_15/Maximum:z:0*
T0*
_output_shapes
:У
EnsureShape_30EnsureShape"UnsortedSegmentMean_15/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         Аф
#update_ip/StatefulPartitionedCall_7StatefulPartitionedCallEnsureShape_30:output:0,update_ip/StatefulPartitionedCall_6:output:0update_ip_4846654update_ip_4846656update_ip_4846658*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_update_ip_layer_call_and_return_conditional_losses_4846653У
EnsureShape_31EnsureShape"UnsortedSegmentMean_14/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АФ
+update_connection/StatefulPartitionedCall_7StatefulPartitionedCallEnsureShape_31:output:04update_connection/StatefulPartitionedCall_6:output:0update_connection_4846702update_connection_4846704update_connection_4846706*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_update_connection_layer_call_and_return_conditional_losses_4846701С
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall4update_connection/StatefulPartitionedCall_7:output:0sequential_2_4847256sequential_2_4847258sequential_2_4847260sequential_2_4847262sequential_2_4847264sequential_2_4847266*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_4845687|
IdentityIdentity-sequential_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         с

NoOpNoOp#^sequential/StatefulPartitionedCall%^sequential/StatefulPartitionedCall_1%^sequential/StatefulPartitionedCall_2%^sequential/StatefulPartitionedCall_3%^sequential/StatefulPartitionedCall_4%^sequential/StatefulPartitionedCall_5%^sequential/StatefulPartitionedCall_6%^sequential/StatefulPartitionedCall_7%^sequential_1/StatefulPartitionedCall'^sequential_1/StatefulPartitionedCall_1'^sequential_1/StatefulPartitionedCall_2'^sequential_1/StatefulPartitionedCall_3'^sequential_1/StatefulPartitionedCall_4'^sequential_1/StatefulPartitionedCall_5'^sequential_1/StatefulPartitionedCall_6'^sequential_1/StatefulPartitionedCall_7%^sequential_2/StatefulPartitionedCall*^update_connection/StatefulPartitionedCall,^update_connection/StatefulPartitionedCall_1,^update_connection/StatefulPartitionedCall_2,^update_connection/StatefulPartitionedCall_3,^update_connection/StatefulPartitionedCall_4,^update_connection/StatefulPartitionedCall_5,^update_connection/StatefulPartitionedCall_6,^update_connection/StatefulPartitionedCall_7"^update_ip/StatefulPartitionedCall$^update_ip/StatefulPartitionedCall_1$^update_ip/StatefulPartitionedCall_2$^update_ip/StatefulPartitionedCall_3$^update_ip/StatefulPartitionedCall_4$^update_ip/StatefulPartitionedCall_5$^update_ip/StatefulPartitionedCall_6$^update_ip/StatefulPartitionedCall_7*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:::: : ::: : : : : : : : : : : : : : : : 2L
$sequential/StatefulPartitionedCall_1$sequential/StatefulPartitionedCall_12L
$sequential/StatefulPartitionedCall_2$sequential/StatefulPartitionedCall_22L
$sequential/StatefulPartitionedCall_3$sequential/StatefulPartitionedCall_32L
$sequential/StatefulPartitionedCall_4$sequential/StatefulPartitionedCall_42L
$sequential/StatefulPartitionedCall_5$sequential/StatefulPartitionedCall_52L
$sequential/StatefulPartitionedCall_6$sequential/StatefulPartitionedCall_62L
$sequential/StatefulPartitionedCall_7$sequential/StatefulPartitionedCall_72H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2P
&sequential_1/StatefulPartitionedCall_1&sequential_1/StatefulPartitionedCall_12P
&sequential_1/StatefulPartitionedCall_2&sequential_1/StatefulPartitionedCall_22P
&sequential_1/StatefulPartitionedCall_3&sequential_1/StatefulPartitionedCall_32P
&sequential_1/StatefulPartitionedCall_4&sequential_1/StatefulPartitionedCall_42P
&sequential_1/StatefulPartitionedCall_5&sequential_1/StatefulPartitionedCall_52P
&sequential_1/StatefulPartitionedCall_6&sequential_1/StatefulPartitionedCall_62P
&sequential_1/StatefulPartitionedCall_7&sequential_1/StatefulPartitionedCall_72L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2Z
+update_connection/StatefulPartitionedCall_1+update_connection/StatefulPartitionedCall_12Z
+update_connection/StatefulPartitionedCall_2+update_connection/StatefulPartitionedCall_22Z
+update_connection/StatefulPartitionedCall_3+update_connection/StatefulPartitionedCall_32Z
+update_connection/StatefulPartitionedCall_4+update_connection/StatefulPartitionedCall_42Z
+update_connection/StatefulPartitionedCall_5+update_connection/StatefulPartitionedCall_52Z
+update_connection/StatefulPartitionedCall_6+update_connection/StatefulPartitionedCall_62Z
+update_connection/StatefulPartitionedCall_7+update_connection/StatefulPartitionedCall_72V
)update_connection/StatefulPartitionedCall)update_connection/StatefulPartitionedCall2J
#update_ip/StatefulPartitionedCall_1#update_ip/StatefulPartitionedCall_12J
#update_ip/StatefulPartitionedCall_2#update_ip/StatefulPartitionedCall_22J
#update_ip/StatefulPartitionedCall_3#update_ip/StatefulPartitionedCall_32J
#update_ip/StatefulPartitionedCall_4#update_ip/StatefulPartitionedCall_42J
#update_ip/StatefulPartitionedCall_5#update_ip/StatefulPartitionedCall_52J
#update_ip/StatefulPartitionedCall_6#update_ip/StatefulPartitionedCall_62J
#update_ip/StatefulPartitionedCall_7#update_ip/StatefulPartitionedCall_72F
!update_ip/StatefulPartitionedCall!update_ip/StatefulPartitionedCall:'#
!
_user_specified_name	4847266:'#
!
_user_specified_name	4847264:'#
!
_user_specified_name	4847262:'#
!
_user_specified_name	4847260:'#
!
_user_specified_name	4847258:'#
!
_user_specified_name	4847256:'#
!
_user_specified_name	4846706:'#
!
_user_specified_name	4846704:'#
!
_user_specified_name	4846702:'#
!
_user_specified_name	4846658:'#
!
_user_specified_name	4846656:'#
!
_user_specified_name	4846654:'
#
!
_user_specified_name	4846590:'	#
!
_user_specified_name	4846588:'#
!
_user_specified_name	4846555:'#
!
_user_specified_name	4846553:NJ

_output_shapes
:
.
_user_specified_namesrc_ip_to_connection:NJ

_output_shapes
:
.
_user_specified_namesrc_connection_to_ip:;7

_output_shapes
: 

_user_specified_namen_i:;7

_output_shapes
: 

_user_specified_namen_c:LH

_output_shapes
:
,
_user_specified_namefeature_connection:NJ

_output_shapes
:
.
_user_specified_namedst_ip_to_connection:N J

_output_shapes
:
.
_user_specified_namedst_connection_to_ip
°
╒
"__inference__wrapped_model_4845408
dst_connection_to_ip	
dst_ip_to_connection	
feature_connection
n_c	
n_i	
src_connection_to_ip	
src_ip_to_connection	
gnn_4845374:
АА
gnn_4845376:	А
gnn_4845378:
АА
gnn_4845380:	А
gnn_4845382:	А
gnn_4845384:
АА
gnn_4845386:
АА
gnn_4845388:	А
gnn_4845390:
АА
gnn_4845392:
АА
gnn_4845394:
АА
gnn_4845396:	А
gnn_4845398:	А@
gnn_4845400:@
gnn_4845402:@
gnn_4845404:
identityИвgnn/StatefulPartitionedCall№
gnn/StatefulPartitionedCallStatefulPartitionedCalldst_connection_to_ipdst_ip_to_connectionfeature_connectionn_cn_isrc_connection_to_ipsrc_ip_to_connectiongnn_4845374gnn_4845376gnn_4845378gnn_4845380gnn_4845382gnn_4845384gnn_4845386gnn_4845388gnn_4845390gnn_4845392gnn_4845394gnn_4845396gnn_4845398gnn_4845400gnn_4845402gnn_4845404*"
Tin
2						*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В * 
fR
__inference_call_557254s
IdentityIdentity$gnn/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @
NoOpNoOp^gnn/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:::: : ::: : : : : : : : : : : : : : : : 2:
gnn/StatefulPartitionedCallgnn/StatefulPartitionedCall:'#
!
_user_specified_name	4845404:'#
!
_user_specified_name	4845402:'#
!
_user_specified_name	4845400:'#
!
_user_specified_name	4845398:'#
!
_user_specified_name	4845396:'#
!
_user_specified_name	4845394:'#
!
_user_specified_name	4845392:'#
!
_user_specified_name	4845390:'#
!
_user_specified_name	4845388:'#
!
_user_specified_name	4845386:'#
!
_user_specified_name	4845384:'#
!
_user_specified_name	4845382:'
#
!
_user_specified_name	4845380:'	#
!
_user_specified_name	4845378:'#
!
_user_specified_name	4845376:'#
!
_user_specified_name	4845374:NJ

_output_shapes
:
.
_user_specified_namesrc_ip_to_connection:NJ

_output_shapes
:
.
_user_specified_namesrc_connection_to_ip:;7

_output_shapes
: 

_user_specified_namen_i:;7

_output_shapes
: 

_user_specified_namen_c:LH

_output_shapes
:
,
_user_specified_namefeature_connection:NJ

_output_shapes
:
.
_user_specified_namedst_ip_to_connection:N J

_output_shapes
:
.
_user_specified_namedst_connection_to_ip
┤
╖
%__inference_signature_wrapper_4847456
dst_connection_to_ip	
dst_ip_to_connection	
feature_connection
n_c	
n_i	
src_connection_to_ip	
src_ip_to_connection	
unknown:
АА
	unknown_0:	А
	unknown_1:
АА
	unknown_2:	А
	unknown_3:	А
	unknown_4:
АА
	unknown_5:
АА
	unknown_6:	А
	unknown_7:
АА
	unknown_8:
АА
	unknown_9:
АА

unknown_10:	А

unknown_11:	А@

unknown_12:@

unknown_13:@

unknown_14:
identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCalldst_connection_to_ipdst_ip_to_connectionfeature_connectionn_cn_isrc_connection_to_ipsrc_ip_to_connectionunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*"
Tin
2						*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__wrapped_model_4845408o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:::: : ::: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	4847452:'#
!
_user_specified_name	4847450:'#
!
_user_specified_name	4847448:'#
!
_user_specified_name	4847446:'#
!
_user_specified_name	4847444:'#
!
_user_specified_name	4847442:'#
!
_user_specified_name	4847440:'#
!
_user_specified_name	4847438:'#
!
_user_specified_name	4847436:'#
!
_user_specified_name	4847434:'#
!
_user_specified_name	4847432:'#
!
_user_specified_name	4847430:'
#
!
_user_specified_name	4847428:'	#
!
_user_specified_name	4847426:'#
!
_user_specified_name	4847424:'#
!
_user_specified_name	4847422:NJ

_output_shapes
:
.
_user_specified_namesrc_ip_to_connection:NJ

_output_shapes
:
.
_user_specified_namesrc_connection_to_ip:;7

_output_shapes
: 

_user_specified_namen_i:;7

_output_shapes
: 

_user_specified_namen_c:LH

_output_shapes
:
,
_user_specified_namefeature_connection:NJ

_output_shapes
:
.
_user_specified_namedst_ip_to_connection:N J

_output_shapes
:
.
_user_specified_namedst_connection_to_ip
б

e
F__inference_dropout_1_layer_call_and_return_conditional_losses_4845507

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╫

°
D__inference_dense_2_layer_call_and_return_conditional_losses_4847782

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         АS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╫

°
D__inference_dense_1_layer_call_and_return_conditional_losses_4845519

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         АS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╨

ї
D__inference_dense_4_layer_call_and_return_conditional_losses_4845649

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ў
Щ
)__inference_dense_1_layer_call_fn_4847751

inputs
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_4845519p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	4847747:'#
!
_user_specified_name	4847745:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Йы	
є%
__inference_call_560424
inputs_dst_connection_to_ip	
inputs_dst_ip_to_connection	
inputs_feature_connection

inputs_n_c	

inputs_n_i	
inputs_src_connection_to_ip	
inputs_src_ip_to_connection	C
/sequential_dense_matmul_readvariableop_resource:
АА?
0sequential_dense_biasadd_readvariableop_resource:	АG
3sequential_1_dense_1_matmul_readvariableop_resource:
ААC
4sequential_1_dense_1_biasadd_readvariableop_resource:	А4
!update_ip_readvariableop_resource:	А<
(update_ip_matmul_readvariableop_resource:
АА>
*update_ip_matmul_1_readvariableop_resource:
АА<
)update_connection_readvariableop_resource:	АD
0update_connection_matmul_readvariableop_resource:
ААF
2update_connection_matmul_1_readvariableop_resource:
ААG
3sequential_2_dense_2_matmul_readvariableop_resource:
ААC
4sequential_2_dense_2_biasadd_readvariableop_resource:	АF
3sequential_2_dense_3_matmul_readvariableop_resource:	А@B
4sequential_2_dense_3_biasadd_readvariableop_resource:@E
3sequential_2_dense_4_matmul_readvariableop_resource:@B
4sequential_2_dense_4_biasadd_readvariableop_resource:
identityИв'sequential/dense/BiasAdd/ReadVariableOpв)sequential/dense/BiasAdd_1/ReadVariableOpв)sequential/dense/BiasAdd_2/ReadVariableOpв)sequential/dense/BiasAdd_3/ReadVariableOpв)sequential/dense/BiasAdd_4/ReadVariableOpв)sequential/dense/BiasAdd_5/ReadVariableOpв)sequential/dense/BiasAdd_6/ReadVariableOpв)sequential/dense/BiasAdd_7/ReadVariableOpв&sequential/dense/MatMul/ReadVariableOpв(sequential/dense/MatMul_1/ReadVariableOpв(sequential/dense/MatMul_2/ReadVariableOpв(sequential/dense/MatMul_3/ReadVariableOpв(sequential/dense/MatMul_4/ReadVariableOpв(sequential/dense/MatMul_5/ReadVariableOpв(sequential/dense/MatMul_6/ReadVariableOpв(sequential/dense/MatMul_7/ReadVariableOpв+sequential_1/dense_1/BiasAdd/ReadVariableOpв-sequential_1/dense_1/BiasAdd_1/ReadVariableOpв-sequential_1/dense_1/BiasAdd_2/ReadVariableOpв-sequential_1/dense_1/BiasAdd_3/ReadVariableOpв-sequential_1/dense_1/BiasAdd_4/ReadVariableOpв-sequential_1/dense_1/BiasAdd_5/ReadVariableOpв-sequential_1/dense_1/BiasAdd_6/ReadVariableOpв-sequential_1/dense_1/BiasAdd_7/ReadVariableOpв*sequential_1/dense_1/MatMul/ReadVariableOpв,sequential_1/dense_1/MatMul_1/ReadVariableOpв,sequential_1/dense_1/MatMul_2/ReadVariableOpв,sequential_1/dense_1/MatMul_3/ReadVariableOpв,sequential_1/dense_1/MatMul_4/ReadVariableOpв,sequential_1/dense_1/MatMul_5/ReadVariableOpв,sequential_1/dense_1/MatMul_6/ReadVariableOpв,sequential_1/dense_1/MatMul_7/ReadVariableOpв+sequential_2/dense_2/BiasAdd/ReadVariableOpв*sequential_2/dense_2/MatMul/ReadVariableOpв+sequential_2/dense_3/BiasAdd/ReadVariableOpв*sequential_2/dense_3/MatMul/ReadVariableOpв+sequential_2/dense_4/BiasAdd/ReadVariableOpв*sequential_2/dense_4/MatMul/ReadVariableOpв'update_connection/MatMul/ReadVariableOpв)update_connection/MatMul_1/ReadVariableOpв*update_connection/MatMul_10/ReadVariableOpв*update_connection/MatMul_11/ReadVariableOpв*update_connection/MatMul_12/ReadVariableOpв*update_connection/MatMul_13/ReadVariableOpв*update_connection/MatMul_14/ReadVariableOpв*update_connection/MatMul_15/ReadVariableOpв)update_connection/MatMul_2/ReadVariableOpв)update_connection/MatMul_3/ReadVariableOpв)update_connection/MatMul_4/ReadVariableOpв)update_connection/MatMul_5/ReadVariableOpв)update_connection/MatMul_6/ReadVariableOpв)update_connection/MatMul_7/ReadVariableOpв)update_connection/MatMul_8/ReadVariableOpв)update_connection/MatMul_9/ReadVariableOpв update_connection/ReadVariableOpв"update_connection/ReadVariableOp_1в"update_connection/ReadVariableOp_2в"update_connection/ReadVariableOp_3в"update_connection/ReadVariableOp_4в"update_connection/ReadVariableOp_5в"update_connection/ReadVariableOp_6в"update_connection/ReadVariableOp_7вupdate_ip/MatMul/ReadVariableOpв!update_ip/MatMul_1/ReadVariableOpв"update_ip/MatMul_10/ReadVariableOpв"update_ip/MatMul_11/ReadVariableOpв"update_ip/MatMul_12/ReadVariableOpв"update_ip/MatMul_13/ReadVariableOpв"update_ip/MatMul_14/ReadVariableOpв"update_ip/MatMul_15/ReadVariableOpв!update_ip/MatMul_2/ReadVariableOpв!update_ip/MatMul_3/ReadVariableOpв!update_ip/MatMul_4/ReadVariableOpв!update_ip/MatMul_5/ReadVariableOpв!update_ip/MatMul_6/ReadVariableOpв!update_ip/MatMul_7/ReadVariableOpв!update_ip/MatMul_8/ReadVariableOpв!update_ip/MatMul_9/ReadVariableOpвupdate_ip/ReadVariableOpвupdate_ip/ReadVariableOp_1вupdate_ip/ReadVariableOp_2вupdate_ip/ReadVariableOp_3вupdate_ip/ReadVariableOp_4вupdate_ip/ReadVariableOp_5вupdate_ip/ReadVariableOp_6вupdate_ip/ReadVariableOp_7P
SqueezeSqueezeinputs_feature_connection*
T0*
_output_shapes
:T
	Squeeze_1Squeezeinputs_src_ip_to_connection*
T0	*
_output_shapes
:T
	Squeeze_2Squeezeinputs_dst_ip_to_connection*
T0	*
_output_shapes
:T
	Squeeze_3Squeezeinputs_src_connection_to_ip*
T0	*
_output_shapes
:T
	Squeeze_4Squeezeinputs_dst_connection_to_ip*
T0	*
_output_shapes
:M
	ones/CastCast
inputs_n_i*

DstT0*

SrcT0	*
_output_shapes
: P
ones/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Аh
ones/packedPackones/Cast:y:0ones/packed/1:output:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?j
onesFillones/packed:output:0ones/Const:output:0*
T0*(
_output_shapes
:         АI
stack/1Const*
_output_shapes
: *
dtype0	*
value	B	 RfY
stackPack
inputs_n_cstack/1:output:0*
N*
T0	*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    w
zerosFillstack:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         f*

index_type0	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :О
concatConcatV2Squeeze:output:0zeros:output:0concat/axis:output:0*
N*
T0*0
_output_shapes
:                  O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Х
GatherV2GatherV2ones:output:0Squeeze_1:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:Q
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ы

GatherV2_1GatherV2concat:output:0Squeeze_2:output:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:L
	Squeeze_5SqueezeGatherV2_1:output:0*
T0*
_output_shapes
:J
	Squeeze_6SqueezeGatherV2:output:0*
T0*
_output_shapes
:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :А
concat_1ConcatV2Squeeze_6:output:0Squeeze_5:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:
EnsureShapeEnsureShapeconcat_1:output:0*
T0*(
_output_shapes
:         А*
shape:         Аp
sequential/dropout/IdentityIdentityEnsureShape:output:0*
T0*(
_output_shapes
:         АШ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0к
sequential/dense/MatMulMatMul$sequential/dropout/Identity:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АХ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0к
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         Аr
UnsortedSegmentMean/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧c
UnsortedSegmentMean/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Р
UnsortedSegmentMean/onesFill"UnsortedSegmentMean/Shape:output:0'UnsortedSegmentMean/ones/Const:output:0*
T0*
_output_shapes
:╞
&UnsortedSegmentMean/UnsortedSegmentSumUnsortedSegmentSum!UnsortedSegmentMean/ones:output:0Squeeze_2:output:0
inputs_n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:q
'UnsortedSegmentMean/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)UnsortedSegmentMean/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: s
)UnsortedSegmentMean/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ю
!UnsortedSegmentMean/strided_sliceStridedSlice
inputs_n_c0UnsortedSegmentMean/strided_slice/stack:output:02UnsortedSegmentMean/strided_slice/stack_1:output:02UnsortedSegmentMean/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_maskZ
UnsortedSegmentMean/RankConst*
_output_shapes
: *
dtype0*
value	B :W
UnsortedSegmentMean/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: З
UnsortedSegmentMean/subSub!UnsortedSegmentMean/Rank:output:0#UnsortedSegmentMean/Rank_1:output:0*
T0*
_output_shapes
: t
!UnsortedSegmentMean/ones_1/packedPackUnsortedSegmentMean/sub:z:0*
N*
T0*
_output_shapes
:b
 UnsortedSegmentMean/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rз
UnsortedSegmentMean/ones_1Fill*UnsortedSegmentMean/ones_1/packed:output:0)UnsortedSegmentMean/ones_1/Const:output:0*
T0	*#
_output_shapes
:         a
UnsortedSegmentMean/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╪
UnsortedSegmentMean/concatConcatV2*UnsortedSegmentMean/strided_slice:output:0#UnsortedSegmentMean/ones_1:output:0(UnsortedSegmentMean/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         н
UnsortedSegmentMean/ReshapeReshape/UnsortedSegmentMean/UnsortedSegmentSum:output:0#UnsortedSegmentMean/concat:output:0*
Tshape0	*
T0*
_output_shapes
:b
UnsortedSegmentMean/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ч
UnsortedSegmentMean/MaximumMaximum$UnsortedSegmentMean/Reshape:output:0&UnsortedSegmentMean/Maximum/y:output:0*
T0*
_output_shapes
:╩
(UnsortedSegmentMean/UnsortedSegmentSum_1UnsortedSegmentSum#sequential/dense/Relu:activations:0Squeeze_2:output:0
inputs_n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:Э
UnsortedSegmentMean/truedivRealDiv1UnsortedSegmentMean/UnsortedSegmentSum_1:output:0UnsortedSegmentMean/Maximum:z:0*
T0*
_output_shapes
:Q
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ы

GatherV2_2GatherV2concat:output:0Squeeze_3:output:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:Q
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : Щ

GatherV2_3GatherV2ones:output:0Squeeze_4:output:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:L
	Squeeze_7SqueezeGatherV2_3:output:0*
T0*
_output_shapes
:L
	Squeeze_8SqueezeGatherV2_2:output:0*
T0*
_output_shapes
:O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :А
concat_2ConcatV2Squeeze_8:output:0Squeeze_7:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:Б
EnsureShape_1EnsureShapeconcat_2:output:0*
T0*(
_output_shapes
:         А*
shape:         Аv
sequential_1/dropout_1/IdentityIdentityEnsureShape_1:output:0*
T0*(
_output_shapes
:         Аа
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╢
sequential_1/dense_1/MatMulMatMul(sequential_1/dropout_1/Identity:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЭ
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╢
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А{
sequential_1/dense_1/ReluRelu%sequential_1/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:         Аt
UnsortedSegmentMean_1/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_1/onesFill$UnsortedSegmentMean_1/Shape:output:0)UnsortedSegmentMean_1/ones/Const:output:0*
T0*
_output_shapes
:╩
(UnsortedSegmentMean_1/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_1/ones:output:0Squeeze_4:output:0
inputs_n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
#UnsortedSegmentMean_1/strided_sliceStridedSlice
inputs_n_i2UnsortedSegmentMean_1/strided_slice/stack:output:04UnsortedSegmentMean_1/strided_slice/stack_1:output:04UnsortedSegmentMean_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_1/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_1/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_1/subSub#UnsortedSegmentMean_1/Rank:output:0%UnsortedSegmentMean_1/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_1/ones_1/packedPackUnsortedSegmentMean_1/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_1/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_1/ones_1Fill,UnsortedSegmentMean_1/ones_1/packed:output:0+UnsortedSegmentMean_1/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_1/concatConcatV2,UnsortedSegmentMean_1/strided_slice:output:0%UnsortedSegmentMean_1/ones_1:output:0*UnsortedSegmentMean_1/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_1/ReshapeReshape1UnsortedSegmentMean_1/UnsortedSegmentSum:output:0%UnsortedSegmentMean_1/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_1/MaximumMaximum&UnsortedSegmentMean_1/Reshape:output:0(UnsortedSegmentMean_1/Maximum/y:output:0*
T0*
_output_shapes
:╨
*UnsortedSegmentMean_1/UnsortedSegmentSum_1UnsortedSegmentSum'sequential_1/dense_1/Relu:activations:0Squeeze_4:output:0
inputs_n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_1/truedivRealDiv3UnsortedSegmentMean_1/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_1/Maximum:z:0*
T0*
_output_shapes
:С
EnsureShape_2EnsureShape!UnsortedSegmentMean_1/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         А{
update_ip/ReadVariableOpReadVariableOp!update_ip_readvariableop_resource*
_output_shapes
:	А*
dtype0u
update_ip/unstackUnpack update_ip/ReadVariableOp:value:0*
T0*"
_output_shapes
:А:А*	
numК
update_ip/MatMul/ReadVariableOpReadVariableOp(update_ip_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0О
update_ip/MatMulMatMulEnsureShape_2:output:0'update_ip/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЗ
update_ip/BiasAddBiasAddupdate_ip/MatMul:product:0update_ip/unstack:output:0*
T0*(
_output_shapes
:         Аd
update_ip/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ─
update_ip/splitSplit"update_ip/split/split_dim:output:0update_ip/BiasAdd:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitО
!update_ip/MatMul_1/ReadVariableOpReadVariableOp*update_ip_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Й
update_ip/MatMul_1MatMulones:output:0)update_ip/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЛ
update_ip/BiasAdd_1BiasAddupdate_ip/MatMul_1:product:0update_ip/unstack:output:1*
T0*(
_output_shapes
:         Аd
update_ip/ConstConst*
_output_shapes
:*
dtype0*!
valueB"А   А       f
update_ip/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ё
update_ip/split_1SplitVupdate_ip/BiasAdd_1:output:0update_ip/Const:output:0$update_ip/split_1/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_split
update_ip/addAddV2update_ip/split:output:0update_ip/split_1:output:0*
T0*(
_output_shapes
:         Аb
update_ip/SigmoidSigmoidupdate_ip/add:z:0*
T0*(
_output_shapes
:         АБ
update_ip/add_1AddV2update_ip/split:output:1update_ip/split_1:output:1*
T0*(
_output_shapes
:         Аf
update_ip/Sigmoid_1Sigmoidupdate_ip/add_1:z:0*
T0*(
_output_shapes
:         А|
update_ip/mulMulupdate_ip/Sigmoid_1:y:0update_ip/split_1:output:2*
T0*(
_output_shapes
:         Аx
update_ip/add_2AddV2update_ip/split:output:2update_ip/mul:z:0*
T0*(
_output_shapes
:         А^
update_ip/TanhTanhupdate_ip/add_2:z:0*
T0*(
_output_shapes
:         Аo
update_ip/mul_1Mulupdate_ip/Sigmoid:y:0ones:output:0*
T0*(
_output_shapes
:         АT
update_ip/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?x
update_ip/subSubupdate_ip/sub/x:output:0update_ip/Sigmoid:y:0*
T0*(
_output_shapes
:         Аp
update_ip/mul_2Mulupdate_ip/sub:z:0update_ip/Tanh:y:0*
T0*(
_output_shapes
:         Аu
update_ip/add_3AddV2update_ip/mul_1:z:0update_ip/mul_2:z:0*
T0*(
_output_shapes
:         АП
EnsureShape_3EnsureShapeUnsortedSegmentMean/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АЛ
 update_connection/ReadVariableOpReadVariableOp)update_connection_readvariableop_resource*
_output_shapes
:	А*
dtype0Е
update_connection/unstackUnpack(update_connection/ReadVariableOp:value:0*
T0*"
_output_shapes
:А:А*	
numЪ
'update_connection/MatMul/ReadVariableOpReadVariableOp0update_connection_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Ю
update_connection/MatMulMatMulEnsureShape_3:output:0/update_connection/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
update_connection/BiasAddBiasAdd"update_connection/MatMul:product:0"update_connection/unstack:output:0*
T0*(
_output_shapes
:         Аl
!update_connection/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ▄
update_connection/splitSplit*update_connection/split/split_dim:output:0"update_connection/BiasAdd:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЮ
)update_connection/MatMul_1/ReadVariableOpReadVariableOp2update_connection_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Ы
update_connection/MatMul_1MatMulconcat:output:01update_connection/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аг
update_connection/BiasAdd_1BiasAdd$update_connection/MatMul_1:product:0"update_connection/unstack:output:1*
T0*(
_output_shapes
:         Аl
update_connection/ConstConst*
_output_shapes
:*
dtype0*!
valueB"А   А       n
#update_connection/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         С
update_connection/split_1SplitV$update_connection/BiasAdd_1:output:0 update_connection/Const:output:0,update_connection/split_1/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЧ
update_connection/addAddV2 update_connection/split:output:0"update_connection/split_1:output:0*
T0*(
_output_shapes
:         Аr
update_connection/SigmoidSigmoidupdate_connection/add:z:0*
T0*(
_output_shapes
:         АЩ
update_connection/add_1AddV2 update_connection/split:output:1"update_connection/split_1:output:1*
T0*(
_output_shapes
:         Аv
update_connection/Sigmoid_1Sigmoidupdate_connection/add_1:z:0*
T0*(
_output_shapes
:         АФ
update_connection/mulMulupdate_connection/Sigmoid_1:y:0"update_connection/split_1:output:2*
T0*(
_output_shapes
:         АР
update_connection/add_2AddV2 update_connection/split:output:2update_connection/mul:z:0*
T0*(
_output_shapes
:         Аn
update_connection/TanhTanhupdate_connection/add_2:z:0*
T0*(
_output_shapes
:         АБ
update_connection/mul_1Mulupdate_connection/Sigmoid:y:0concat:output:0*
T0*(
_output_shapes
:         А\
update_connection/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Р
update_connection/subSub update_connection/sub/x:output:0update_connection/Sigmoid:y:0*
T0*(
_output_shapes
:         АИ
update_connection/mul_2Mulupdate_connection/sub:z:0update_connection/Tanh:y:0*
T0*(
_output_shapes
:         АН
update_connection/add_3AddV2update_connection/mul_1:z:0update_connection/mul_2:z:0*
T0*(
_output_shapes
:         АQ
GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B : Я

GatherV2_4GatherV2update_ip/add_3:z:0Squeeze_1:output:0GatherV2_4/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:Q
GatherV2_5/axisConst*
_output_shapes
: *
dtype0*
value	B : з

GatherV2_5GatherV2update_connection/add_3:z:0Squeeze_2:output:0GatherV2_5/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:L
	Squeeze_9SqueezeGatherV2_5:output:0*
T0*
_output_shapes
:M

Squeeze_10SqueezeGatherV2_4:output:0*
T0*
_output_shapes
:O
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B :Б
concat_3ConcatV2Squeeze_10:output:0Squeeze_9:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:Б
EnsureShape_4EnsureShapeconcat_3:output:0*
T0*(
_output_shapes
:         А*
shape:         Аt
sequential/dropout/Identity_1IdentityEnsureShape_4:output:0*
T0*(
_output_shapes
:         АЪ
(sequential/dense/MatMul_1/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0░
sequential/dense/MatMul_1MatMul&sequential/dropout/Identity_1:output:00sequential/dense/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЧ
)sequential/dense/BiasAdd_1/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0░
sequential/dense/BiasAdd_1BiasAdd#sequential/dense/MatMul_1:product:01sequential/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аw
sequential/dense/Relu_1Relu#sequential/dense/BiasAdd_1:output:0*
T0*(
_output_shapes
:         Аt
UnsortedSegmentMean_2/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_2/onesFill$UnsortedSegmentMean_2/Shape:output:0)UnsortedSegmentMean_2/ones/Const:output:0*
T0*
_output_shapes
:╩
(UnsortedSegmentMean_2/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_2/ones:output:0Squeeze_2:output:0
inputs_n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
#UnsortedSegmentMean_2/strided_sliceStridedSlice
inputs_n_c2UnsortedSegmentMean_2/strided_slice/stack:output:04UnsortedSegmentMean_2/strided_slice/stack_1:output:04UnsortedSegmentMean_2/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_2/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_2/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_2/subSub#UnsortedSegmentMean_2/Rank:output:0%UnsortedSegmentMean_2/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_2/ones_1/packedPackUnsortedSegmentMean_2/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_2/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_2/ones_1Fill,UnsortedSegmentMean_2/ones_1/packed:output:0+UnsortedSegmentMean_2/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_2/concatConcatV2,UnsortedSegmentMean_2/strided_slice:output:0%UnsortedSegmentMean_2/ones_1:output:0*UnsortedSegmentMean_2/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_2/ReshapeReshape1UnsortedSegmentMean_2/UnsortedSegmentSum:output:0%UnsortedSegmentMean_2/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_2/MaximumMaximum&UnsortedSegmentMean_2/Reshape:output:0(UnsortedSegmentMean_2/Maximum/y:output:0*
T0*
_output_shapes
:╬
*UnsortedSegmentMean_2/UnsortedSegmentSum_1UnsortedSegmentSum%sequential/dense/Relu_1:activations:0Squeeze_2:output:0
inputs_n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_2/truedivRealDiv3UnsortedSegmentMean_2/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_2/Maximum:z:0*
T0*
_output_shapes
:Q
GatherV2_6/axisConst*
_output_shapes
: *
dtype0*
value	B : з

GatherV2_6GatherV2update_connection/add_3:z:0Squeeze_3:output:0GatherV2_6/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:Q
GatherV2_7/axisConst*
_output_shapes
: *
dtype0*
value	B : Я

GatherV2_7GatherV2update_ip/add_3:z:0Squeeze_4:output:0GatherV2_7/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:M

Squeeze_11SqueezeGatherV2_7:output:0*
T0*
_output_shapes
:M

Squeeze_12SqueezeGatherV2_6:output:0*
T0*
_output_shapes
:O
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concat_4ConcatV2Squeeze_12:output:0Squeeze_11:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:Б
EnsureShape_5EnsureShapeconcat_4:output:0*
T0*(
_output_shapes
:         А*
shape:         Аx
!sequential_1/dropout_1/Identity_1IdentityEnsureShape_5:output:0*
T0*(
_output_shapes
:         Ав
,sequential_1/dense_1/MatMul_1/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╝
sequential_1/dense_1/MatMul_1MatMul*sequential_1/dropout_1/Identity_1:output:04sequential_1/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
-sequential_1/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╝
sequential_1/dense_1/BiasAdd_1BiasAdd'sequential_1/dense_1/MatMul_1:product:05sequential_1/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А
sequential_1/dense_1/Relu_1Relu'sequential_1/dense_1/BiasAdd_1:output:0*
T0*(
_output_shapes
:         Аt
UnsortedSegmentMean_3/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_3/onesFill$UnsortedSegmentMean_3/Shape:output:0)UnsortedSegmentMean_3/ones/Const:output:0*
T0*
_output_shapes
:╩
(UnsortedSegmentMean_3/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_3/ones:output:0Squeeze_4:output:0
inputs_n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
#UnsortedSegmentMean_3/strided_sliceStridedSlice
inputs_n_i2UnsortedSegmentMean_3/strided_slice/stack:output:04UnsortedSegmentMean_3/strided_slice/stack_1:output:04UnsortedSegmentMean_3/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_3/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_3/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_3/subSub#UnsortedSegmentMean_3/Rank:output:0%UnsortedSegmentMean_3/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_3/ones_1/packedPackUnsortedSegmentMean_3/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_3/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_3/ones_1Fill,UnsortedSegmentMean_3/ones_1/packed:output:0+UnsortedSegmentMean_3/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_3/concatConcatV2,UnsortedSegmentMean_3/strided_slice:output:0%UnsortedSegmentMean_3/ones_1:output:0*UnsortedSegmentMean_3/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_3/ReshapeReshape1UnsortedSegmentMean_3/UnsortedSegmentSum:output:0%UnsortedSegmentMean_3/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_3/MaximumMaximum&UnsortedSegmentMean_3/Reshape:output:0(UnsortedSegmentMean_3/Maximum/y:output:0*
T0*
_output_shapes
:╥
*UnsortedSegmentMean_3/UnsortedSegmentSum_1UnsortedSegmentSum)sequential_1/dense_1/Relu_1:activations:0Squeeze_4:output:0
inputs_n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_3/truedivRealDiv3UnsortedSegmentMean_3/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_3/Maximum:z:0*
T0*
_output_shapes
:С
EnsureShape_6EnsureShape!UnsortedSegmentMean_3/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         А}
update_ip/ReadVariableOp_1ReadVariableOp!update_ip_readvariableop_resource*
_output_shapes
:	А*
dtype0y
update_ip/unstack_1Unpack"update_ip/ReadVariableOp_1:value:0*
T0*"
_output_shapes
:А:А*	
numМ
!update_ip/MatMul_2/ReadVariableOpReadVariableOp(update_ip_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Т
update_ip/MatMul_2MatMulEnsureShape_6:output:0)update_ip/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АН
update_ip/BiasAdd_2BiasAddupdate_ip/MatMul_2:product:0update_ip/unstack_1:output:0*
T0*(
_output_shapes
:         Аf
update_ip/split_2/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╩
update_ip/split_2Split$update_ip/split_2/split_dim:output:0update_ip/BiasAdd_2:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitО
!update_ip/MatMul_3/ReadVariableOpReadVariableOp*update_ip_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0П
update_ip/MatMul_3MatMulupdate_ip/add_3:z:0)update_ip/MatMul_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АН
update_ip/BiasAdd_3BiasAddupdate_ip/MatMul_3:product:0update_ip/unstack_1:output:1*
T0*(
_output_shapes
:         Аf
update_ip/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"А   А       f
update_ip/split_3/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         є
update_ip/split_3SplitVupdate_ip/BiasAdd_3:output:0update_ip/Const_1:output:0$update_ip/split_3/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitГ
update_ip/add_4AddV2update_ip/split_2:output:0update_ip/split_3:output:0*
T0*(
_output_shapes
:         Аf
update_ip/Sigmoid_2Sigmoidupdate_ip/add_4:z:0*
T0*(
_output_shapes
:         АГ
update_ip/add_5AddV2update_ip/split_2:output:1update_ip/split_3:output:1*
T0*(
_output_shapes
:         Аf
update_ip/Sigmoid_3Sigmoidupdate_ip/add_5:z:0*
T0*(
_output_shapes
:         А~
update_ip/mul_3Mulupdate_ip/Sigmoid_3:y:0update_ip/split_3:output:2*
T0*(
_output_shapes
:         А|
update_ip/add_6AddV2update_ip/split_2:output:2update_ip/mul_3:z:0*
T0*(
_output_shapes
:         А`
update_ip/Tanh_1Tanhupdate_ip/add_6:z:0*
T0*(
_output_shapes
:         Аw
update_ip/mul_4Mulupdate_ip/Sigmoid_2:y:0update_ip/add_3:z:0*
T0*(
_output_shapes
:         АV
update_ip/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?~
update_ip/sub_1Subupdate_ip/sub_1/x:output:0update_ip/Sigmoid_2:y:0*
T0*(
_output_shapes
:         Аt
update_ip/mul_5Mulupdate_ip/sub_1:z:0update_ip/Tanh_1:y:0*
T0*(
_output_shapes
:         Аu
update_ip/add_7AddV2update_ip/mul_4:z:0update_ip/mul_5:z:0*
T0*(
_output_shapes
:         АС
EnsureShape_7EnsureShape!UnsortedSegmentMean_2/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АН
"update_connection/ReadVariableOp_1ReadVariableOp)update_connection_readvariableop_resource*
_output_shapes
:	А*
dtype0Й
update_connection/unstack_1Unpack*update_connection/ReadVariableOp_1:value:0*
T0*"
_output_shapes
:А:А*	
numЬ
)update_connection/MatMul_2/ReadVariableOpReadVariableOp0update_connection_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0в
update_connection/MatMul_2MatMulEnsureShape_7:output:01update_connection/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ае
update_connection/BiasAdd_2BiasAdd$update_connection/MatMul_2:product:0$update_connection/unstack_1:output:0*
T0*(
_output_shapes
:         Аn
#update_connection/split_2/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         т
update_connection/split_2Split,update_connection/split_2/split_dim:output:0$update_connection/BiasAdd_2:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЮ
)update_connection/MatMul_3/ReadVariableOpReadVariableOp2update_connection_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0з
update_connection/MatMul_3MatMulupdate_connection/add_3:z:01update_connection/MatMul_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ае
update_connection/BiasAdd_3BiasAdd$update_connection/MatMul_3:product:0$update_connection/unstack_1:output:1*
T0*(
_output_shapes
:         Аn
update_connection/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"А   А       n
#update_connection/split_3/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         У
update_connection/split_3SplitV$update_connection/BiasAdd_3:output:0"update_connection/Const_1:output:0,update_connection/split_3/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЫ
update_connection/add_4AddV2"update_connection/split_2:output:0"update_connection/split_3:output:0*
T0*(
_output_shapes
:         Аv
update_connection/Sigmoid_2Sigmoidupdate_connection/add_4:z:0*
T0*(
_output_shapes
:         АЫ
update_connection/add_5AddV2"update_connection/split_2:output:1"update_connection/split_3:output:1*
T0*(
_output_shapes
:         Аv
update_connection/Sigmoid_3Sigmoidupdate_connection/add_5:z:0*
T0*(
_output_shapes
:         АЦ
update_connection/mul_3Mulupdate_connection/Sigmoid_3:y:0"update_connection/split_3:output:2*
T0*(
_output_shapes
:         АФ
update_connection/add_6AddV2"update_connection/split_2:output:2update_connection/mul_3:z:0*
T0*(
_output_shapes
:         Аp
update_connection/Tanh_1Tanhupdate_connection/add_6:z:0*
T0*(
_output_shapes
:         АП
update_connection/mul_4Mulupdate_connection/Sigmoid_2:y:0update_connection/add_3:z:0*
T0*(
_output_shapes
:         А^
update_connection/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
update_connection/sub_1Sub"update_connection/sub_1/x:output:0update_connection/Sigmoid_2:y:0*
T0*(
_output_shapes
:         АМ
update_connection/mul_5Mulupdate_connection/sub_1:z:0update_connection/Tanh_1:y:0*
T0*(
_output_shapes
:         АН
update_connection/add_7AddV2update_connection/mul_4:z:0update_connection/mul_5:z:0*
T0*(
_output_shapes
:         АQ
GatherV2_8/axisConst*
_output_shapes
: *
dtype0*
value	B : Я

GatherV2_8GatherV2update_ip/add_7:z:0Squeeze_1:output:0GatherV2_8/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:Q
GatherV2_9/axisConst*
_output_shapes
: *
dtype0*
value	B : з

GatherV2_9GatherV2update_connection/add_7:z:0Squeeze_2:output:0GatherV2_9/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:M

Squeeze_13SqueezeGatherV2_9:output:0*
T0*
_output_shapes
:M

Squeeze_14SqueezeGatherV2_8:output:0*
T0*
_output_shapes
:O
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concat_5ConcatV2Squeeze_14:output:0Squeeze_13:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes
:Б
EnsureShape_8EnsureShapeconcat_5:output:0*
T0*(
_output_shapes
:         А*
shape:         Аt
sequential/dropout/Identity_2IdentityEnsureShape_8:output:0*
T0*(
_output_shapes
:         АЪ
(sequential/dense/MatMul_2/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0░
sequential/dense/MatMul_2MatMul&sequential/dropout/Identity_2:output:00sequential/dense/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЧ
)sequential/dense/BiasAdd_2/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0░
sequential/dense/BiasAdd_2BiasAdd#sequential/dense/MatMul_2:product:01sequential/dense/BiasAdd_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аw
sequential/dense/Relu_2Relu#sequential/dense/BiasAdd_2:output:0*
T0*(
_output_shapes
:         Аt
UnsortedSegmentMean_4/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_4/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_4/onesFill$UnsortedSegmentMean_4/Shape:output:0)UnsortedSegmentMean_4/ones/Const:output:0*
T0*
_output_shapes
:╩
(UnsortedSegmentMean_4/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_4/ones:output:0Squeeze_2:output:0
inputs_n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
#UnsortedSegmentMean_4/strided_sliceStridedSlice
inputs_n_c2UnsortedSegmentMean_4/strided_slice/stack:output:04UnsortedSegmentMean_4/strided_slice/stack_1:output:04UnsortedSegmentMean_4/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_4/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_4/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_4/subSub#UnsortedSegmentMean_4/Rank:output:0%UnsortedSegmentMean_4/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_4/ones_1/packedPackUnsortedSegmentMean_4/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_4/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_4/ones_1Fill,UnsortedSegmentMean_4/ones_1/packed:output:0+UnsortedSegmentMean_4/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_4/concatConcatV2,UnsortedSegmentMean_4/strided_slice:output:0%UnsortedSegmentMean_4/ones_1:output:0*UnsortedSegmentMean_4/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_4/ReshapeReshape1UnsortedSegmentMean_4/UnsortedSegmentSum:output:0%UnsortedSegmentMean_4/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_4/MaximumMaximum&UnsortedSegmentMean_4/Reshape:output:0(UnsortedSegmentMean_4/Maximum/y:output:0*
T0*
_output_shapes
:╬
*UnsortedSegmentMean_4/UnsortedSegmentSum_1UnsortedSegmentSum%sequential/dense/Relu_2:activations:0Squeeze_2:output:0
inputs_n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_4/truedivRealDiv3UnsortedSegmentMean_4/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_4/Maximum:z:0*
T0*
_output_shapes
:R
GatherV2_10/axisConst*
_output_shapes
: *
dtype0*
value	B : й
GatherV2_10GatherV2update_connection/add_7:z:0Squeeze_3:output:0GatherV2_10/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_11/axisConst*
_output_shapes
: *
dtype0*
value	B : б
GatherV2_11GatherV2update_ip/add_7:z:0Squeeze_4:output:0GatherV2_11/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_15SqueezeGatherV2_11:output:0*
T0*
_output_shapes
:N

Squeeze_16SqueezeGatherV2_10:output:0*
T0*
_output_shapes
:O
concat_6/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concat_6ConcatV2Squeeze_16:output:0Squeeze_15:output:0concat_6/axis:output:0*
N*
T0*
_output_shapes
:Б
EnsureShape_9EnsureShapeconcat_6:output:0*
T0*(
_output_shapes
:         А*
shape:         Аx
!sequential_1/dropout_1/Identity_2IdentityEnsureShape_9:output:0*
T0*(
_output_shapes
:         Ав
,sequential_1/dense_1/MatMul_2/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╝
sequential_1/dense_1/MatMul_2MatMul*sequential_1/dropout_1/Identity_2:output:04sequential_1/dense_1/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
-sequential_1/dense_1/BiasAdd_2/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╝
sequential_1/dense_1/BiasAdd_2BiasAdd'sequential_1/dense_1/MatMul_2:product:05sequential_1/dense_1/BiasAdd_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А
sequential_1/dense_1/Relu_2Relu'sequential_1/dense_1/BiasAdd_2:output:0*
T0*(
_output_shapes
:         Аt
UnsortedSegmentMean_5/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_5/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_5/onesFill$UnsortedSegmentMean_5/Shape:output:0)UnsortedSegmentMean_5/ones/Const:output:0*
T0*
_output_shapes
:╩
(UnsortedSegmentMean_5/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_5/ones:output:0Squeeze_4:output:0
inputs_n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
#UnsortedSegmentMean_5/strided_sliceStridedSlice
inputs_n_i2UnsortedSegmentMean_5/strided_slice/stack:output:04UnsortedSegmentMean_5/strided_slice/stack_1:output:04UnsortedSegmentMean_5/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_5/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_5/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_5/subSub#UnsortedSegmentMean_5/Rank:output:0%UnsortedSegmentMean_5/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_5/ones_1/packedPackUnsortedSegmentMean_5/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_5/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_5/ones_1Fill,UnsortedSegmentMean_5/ones_1/packed:output:0+UnsortedSegmentMean_5/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_5/concatConcatV2,UnsortedSegmentMean_5/strided_slice:output:0%UnsortedSegmentMean_5/ones_1:output:0*UnsortedSegmentMean_5/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_5/ReshapeReshape1UnsortedSegmentMean_5/UnsortedSegmentSum:output:0%UnsortedSegmentMean_5/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_5/MaximumMaximum&UnsortedSegmentMean_5/Reshape:output:0(UnsortedSegmentMean_5/Maximum/y:output:0*
T0*
_output_shapes
:╥
*UnsortedSegmentMean_5/UnsortedSegmentSum_1UnsortedSegmentSum)sequential_1/dense_1/Relu_2:activations:0Squeeze_4:output:0
inputs_n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_5/truedivRealDiv3UnsortedSegmentMean_5/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_5/Maximum:z:0*
T0*
_output_shapes
:Т
EnsureShape_10EnsureShape!UnsortedSegmentMean_5/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         А}
update_ip/ReadVariableOp_2ReadVariableOp!update_ip_readvariableop_resource*
_output_shapes
:	А*
dtype0y
update_ip/unstack_2Unpack"update_ip/ReadVariableOp_2:value:0*
T0*"
_output_shapes
:А:А*	
numМ
!update_ip/MatMul_4/ReadVariableOpReadVariableOp(update_ip_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0У
update_ip/MatMul_4MatMulEnsureShape_10:output:0)update_ip/MatMul_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АН
update_ip/BiasAdd_4BiasAddupdate_ip/MatMul_4:product:0update_ip/unstack_2:output:0*
T0*(
_output_shapes
:         Аf
update_ip/split_4/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╩
update_ip/split_4Split$update_ip/split_4/split_dim:output:0update_ip/BiasAdd_4:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitО
!update_ip/MatMul_5/ReadVariableOpReadVariableOp*update_ip_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0П
update_ip/MatMul_5MatMulupdate_ip/add_7:z:0)update_ip/MatMul_5/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АН
update_ip/BiasAdd_5BiasAddupdate_ip/MatMul_5:product:0update_ip/unstack_2:output:1*
T0*(
_output_shapes
:         Аf
update_ip/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"А   А       f
update_ip/split_5/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         є
update_ip/split_5SplitVupdate_ip/BiasAdd_5:output:0update_ip/Const_2:output:0$update_ip/split_5/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitГ
update_ip/add_8AddV2update_ip/split_4:output:0update_ip/split_5:output:0*
T0*(
_output_shapes
:         Аf
update_ip/Sigmoid_4Sigmoidupdate_ip/add_8:z:0*
T0*(
_output_shapes
:         АГ
update_ip/add_9AddV2update_ip/split_4:output:1update_ip/split_5:output:1*
T0*(
_output_shapes
:         Аf
update_ip/Sigmoid_5Sigmoidupdate_ip/add_9:z:0*
T0*(
_output_shapes
:         А~
update_ip/mul_6Mulupdate_ip/Sigmoid_5:y:0update_ip/split_5:output:2*
T0*(
_output_shapes
:         А}
update_ip/add_10AddV2update_ip/split_4:output:2update_ip/mul_6:z:0*
T0*(
_output_shapes
:         Аa
update_ip/Tanh_2Tanhupdate_ip/add_10:z:0*
T0*(
_output_shapes
:         Аw
update_ip/mul_7Mulupdate_ip/Sigmoid_4:y:0update_ip/add_7:z:0*
T0*(
_output_shapes
:         АV
update_ip/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?~
update_ip/sub_2Subupdate_ip/sub_2/x:output:0update_ip/Sigmoid_4:y:0*
T0*(
_output_shapes
:         Аt
update_ip/mul_8Mulupdate_ip/sub_2:z:0update_ip/Tanh_2:y:0*
T0*(
_output_shapes
:         Аv
update_ip/add_11AddV2update_ip/mul_7:z:0update_ip/mul_8:z:0*
T0*(
_output_shapes
:         АТ
EnsureShape_11EnsureShape!UnsortedSegmentMean_4/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АН
"update_connection/ReadVariableOp_2ReadVariableOp)update_connection_readvariableop_resource*
_output_shapes
:	А*
dtype0Й
update_connection/unstack_2Unpack*update_connection/ReadVariableOp_2:value:0*
T0*"
_output_shapes
:А:А*	
numЬ
)update_connection/MatMul_4/ReadVariableOpReadVariableOp0update_connection_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0г
update_connection/MatMul_4MatMulEnsureShape_11:output:01update_connection/MatMul_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ае
update_connection/BiasAdd_4BiasAdd$update_connection/MatMul_4:product:0$update_connection/unstack_2:output:0*
T0*(
_output_shapes
:         Аn
#update_connection/split_4/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         т
update_connection/split_4Split,update_connection/split_4/split_dim:output:0$update_connection/BiasAdd_4:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЮ
)update_connection/MatMul_5/ReadVariableOpReadVariableOp2update_connection_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0з
update_connection/MatMul_5MatMulupdate_connection/add_7:z:01update_connection/MatMul_5/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ае
update_connection/BiasAdd_5BiasAdd$update_connection/MatMul_5:product:0$update_connection/unstack_2:output:1*
T0*(
_output_shapes
:         Аn
update_connection/Const_2Const*
_output_shapes
:*
dtype0*!
valueB"А   А       n
#update_connection/split_5/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         У
update_connection/split_5SplitV$update_connection/BiasAdd_5:output:0"update_connection/Const_2:output:0,update_connection/split_5/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЫ
update_connection/add_8AddV2"update_connection/split_4:output:0"update_connection/split_5:output:0*
T0*(
_output_shapes
:         Аv
update_connection/Sigmoid_4Sigmoidupdate_connection/add_8:z:0*
T0*(
_output_shapes
:         АЫ
update_connection/add_9AddV2"update_connection/split_4:output:1"update_connection/split_5:output:1*
T0*(
_output_shapes
:         Аv
update_connection/Sigmoid_5Sigmoidupdate_connection/add_9:z:0*
T0*(
_output_shapes
:         АЦ
update_connection/mul_6Mulupdate_connection/Sigmoid_5:y:0"update_connection/split_5:output:2*
T0*(
_output_shapes
:         АХ
update_connection/add_10AddV2"update_connection/split_4:output:2update_connection/mul_6:z:0*
T0*(
_output_shapes
:         Аq
update_connection/Tanh_2Tanhupdate_connection/add_10:z:0*
T0*(
_output_shapes
:         АП
update_connection/mul_7Mulupdate_connection/Sigmoid_4:y:0update_connection/add_7:z:0*
T0*(
_output_shapes
:         А^
update_connection/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
update_connection/sub_2Sub"update_connection/sub_2/x:output:0update_connection/Sigmoid_4:y:0*
T0*(
_output_shapes
:         АМ
update_connection/mul_8Mulupdate_connection/sub_2:z:0update_connection/Tanh_2:y:0*
T0*(
_output_shapes
:         АО
update_connection/add_11AddV2update_connection/mul_7:z:0update_connection/mul_8:z:0*
T0*(
_output_shapes
:         АR
GatherV2_12/axisConst*
_output_shapes
: *
dtype0*
value	B : в
GatherV2_12GatherV2update_ip/add_11:z:0Squeeze_1:output:0GatherV2_12/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_13/axisConst*
_output_shapes
: *
dtype0*
value	B : к
GatherV2_13GatherV2update_connection/add_11:z:0Squeeze_2:output:0GatherV2_13/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_17SqueezeGatherV2_13:output:0*
T0*
_output_shapes
:N

Squeeze_18SqueezeGatherV2_12:output:0*
T0*
_output_shapes
:O
concat_7/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concat_7ConcatV2Squeeze_18:output:0Squeeze_17:output:0concat_7/axis:output:0*
N*
T0*
_output_shapes
:В
EnsureShape_12EnsureShapeconcat_7:output:0*
T0*(
_output_shapes
:         А*
shape:         Аu
sequential/dropout/Identity_3IdentityEnsureShape_12:output:0*
T0*(
_output_shapes
:         АЪ
(sequential/dense/MatMul_3/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0░
sequential/dense/MatMul_3MatMul&sequential/dropout/Identity_3:output:00sequential/dense/MatMul_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЧ
)sequential/dense/BiasAdd_3/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0░
sequential/dense/BiasAdd_3BiasAdd#sequential/dense/MatMul_3:product:01sequential/dense/BiasAdd_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аw
sequential/dense/Relu_3Relu#sequential/dense/BiasAdd_3:output:0*
T0*(
_output_shapes
:         Аt
UnsortedSegmentMean_6/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_6/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_6/onesFill$UnsortedSegmentMean_6/Shape:output:0)UnsortedSegmentMean_6/ones/Const:output:0*
T0*
_output_shapes
:╩
(UnsortedSegmentMean_6/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_6/ones:output:0Squeeze_2:output:0
inputs_n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
#UnsortedSegmentMean_6/strided_sliceStridedSlice
inputs_n_c2UnsortedSegmentMean_6/strided_slice/stack:output:04UnsortedSegmentMean_6/strided_slice/stack_1:output:04UnsortedSegmentMean_6/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_6/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_6/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_6/subSub#UnsortedSegmentMean_6/Rank:output:0%UnsortedSegmentMean_6/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_6/ones_1/packedPackUnsortedSegmentMean_6/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_6/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_6/ones_1Fill,UnsortedSegmentMean_6/ones_1/packed:output:0+UnsortedSegmentMean_6/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_6/concatConcatV2,UnsortedSegmentMean_6/strided_slice:output:0%UnsortedSegmentMean_6/ones_1:output:0*UnsortedSegmentMean_6/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_6/ReshapeReshape1UnsortedSegmentMean_6/UnsortedSegmentSum:output:0%UnsortedSegmentMean_6/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_6/MaximumMaximum&UnsortedSegmentMean_6/Reshape:output:0(UnsortedSegmentMean_6/Maximum/y:output:0*
T0*
_output_shapes
:╬
*UnsortedSegmentMean_6/UnsortedSegmentSum_1UnsortedSegmentSum%sequential/dense/Relu_3:activations:0Squeeze_2:output:0
inputs_n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_6/truedivRealDiv3UnsortedSegmentMean_6/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_6/Maximum:z:0*
T0*
_output_shapes
:R
GatherV2_14/axisConst*
_output_shapes
: *
dtype0*
value	B : к
GatherV2_14GatherV2update_connection/add_11:z:0Squeeze_3:output:0GatherV2_14/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_15/axisConst*
_output_shapes
: *
dtype0*
value	B : в
GatherV2_15GatherV2update_ip/add_11:z:0Squeeze_4:output:0GatherV2_15/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_19SqueezeGatherV2_15:output:0*
T0*
_output_shapes
:N

Squeeze_20SqueezeGatherV2_14:output:0*
T0*
_output_shapes
:O
concat_8/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concat_8ConcatV2Squeeze_20:output:0Squeeze_19:output:0concat_8/axis:output:0*
N*
T0*
_output_shapes
:В
EnsureShape_13EnsureShapeconcat_8:output:0*
T0*(
_output_shapes
:         А*
shape:         Аy
!sequential_1/dropout_1/Identity_3IdentityEnsureShape_13:output:0*
T0*(
_output_shapes
:         Ав
,sequential_1/dense_1/MatMul_3/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╝
sequential_1/dense_1/MatMul_3MatMul*sequential_1/dropout_1/Identity_3:output:04sequential_1/dense_1/MatMul_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
-sequential_1/dense_1/BiasAdd_3/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╝
sequential_1/dense_1/BiasAdd_3BiasAdd'sequential_1/dense_1/MatMul_3:product:05sequential_1/dense_1/BiasAdd_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А
sequential_1/dense_1/Relu_3Relu'sequential_1/dense_1/BiasAdd_3:output:0*
T0*(
_output_shapes
:         Аt
UnsortedSegmentMean_7/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_7/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_7/onesFill$UnsortedSegmentMean_7/Shape:output:0)UnsortedSegmentMean_7/ones/Const:output:0*
T0*
_output_shapes
:╩
(UnsortedSegmentMean_7/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_7/ones:output:0Squeeze_4:output:0
inputs_n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
#UnsortedSegmentMean_7/strided_sliceStridedSlice
inputs_n_i2UnsortedSegmentMean_7/strided_slice/stack:output:04UnsortedSegmentMean_7/strided_slice/stack_1:output:04UnsortedSegmentMean_7/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_7/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_7/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_7/subSub#UnsortedSegmentMean_7/Rank:output:0%UnsortedSegmentMean_7/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_7/ones_1/packedPackUnsortedSegmentMean_7/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_7/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_7/ones_1Fill,UnsortedSegmentMean_7/ones_1/packed:output:0+UnsortedSegmentMean_7/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_7/concatConcatV2,UnsortedSegmentMean_7/strided_slice:output:0%UnsortedSegmentMean_7/ones_1:output:0*UnsortedSegmentMean_7/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_7/ReshapeReshape1UnsortedSegmentMean_7/UnsortedSegmentSum:output:0%UnsortedSegmentMean_7/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_7/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_7/MaximumMaximum&UnsortedSegmentMean_7/Reshape:output:0(UnsortedSegmentMean_7/Maximum/y:output:0*
T0*
_output_shapes
:╥
*UnsortedSegmentMean_7/UnsortedSegmentSum_1UnsortedSegmentSum)sequential_1/dense_1/Relu_3:activations:0Squeeze_4:output:0
inputs_n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_7/truedivRealDiv3UnsortedSegmentMean_7/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_7/Maximum:z:0*
T0*
_output_shapes
:Т
EnsureShape_14EnsureShape!UnsortedSegmentMean_7/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         А}
update_ip/ReadVariableOp_3ReadVariableOp!update_ip_readvariableop_resource*
_output_shapes
:	А*
dtype0y
update_ip/unstack_3Unpack"update_ip/ReadVariableOp_3:value:0*
T0*"
_output_shapes
:А:А*	
numМ
!update_ip/MatMul_6/ReadVariableOpReadVariableOp(update_ip_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0У
update_ip/MatMul_6MatMulEnsureShape_14:output:0)update_ip/MatMul_6/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АН
update_ip/BiasAdd_6BiasAddupdate_ip/MatMul_6:product:0update_ip/unstack_3:output:0*
T0*(
_output_shapes
:         Аf
update_ip/split_6/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╩
update_ip/split_6Split$update_ip/split_6/split_dim:output:0update_ip/BiasAdd_6:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitО
!update_ip/MatMul_7/ReadVariableOpReadVariableOp*update_ip_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Р
update_ip/MatMul_7MatMulupdate_ip/add_11:z:0)update_ip/MatMul_7/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АН
update_ip/BiasAdd_7BiasAddupdate_ip/MatMul_7:product:0update_ip/unstack_3:output:1*
T0*(
_output_shapes
:         Аf
update_ip/Const_3Const*
_output_shapes
:*
dtype0*!
valueB"А   А       f
update_ip/split_7/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         є
update_ip/split_7SplitVupdate_ip/BiasAdd_7:output:0update_ip/Const_3:output:0$update_ip/split_7/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitД
update_ip/add_12AddV2update_ip/split_6:output:0update_ip/split_7:output:0*
T0*(
_output_shapes
:         Аg
update_ip/Sigmoid_6Sigmoidupdate_ip/add_12:z:0*
T0*(
_output_shapes
:         АД
update_ip/add_13AddV2update_ip/split_6:output:1update_ip/split_7:output:1*
T0*(
_output_shapes
:         Аg
update_ip/Sigmoid_7Sigmoidupdate_ip/add_13:z:0*
T0*(
_output_shapes
:         А~
update_ip/mul_9Mulupdate_ip/Sigmoid_7:y:0update_ip/split_7:output:2*
T0*(
_output_shapes
:         А}
update_ip/add_14AddV2update_ip/split_6:output:2update_ip/mul_9:z:0*
T0*(
_output_shapes
:         Аa
update_ip/Tanh_3Tanhupdate_ip/add_14:z:0*
T0*(
_output_shapes
:         Аy
update_ip/mul_10Mulupdate_ip/Sigmoid_6:y:0update_ip/add_11:z:0*
T0*(
_output_shapes
:         АV
update_ip/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?~
update_ip/sub_3Subupdate_ip/sub_3/x:output:0update_ip/Sigmoid_6:y:0*
T0*(
_output_shapes
:         Аu
update_ip/mul_11Mulupdate_ip/sub_3:z:0update_ip/Tanh_3:y:0*
T0*(
_output_shapes
:         Аx
update_ip/add_15AddV2update_ip/mul_10:z:0update_ip/mul_11:z:0*
T0*(
_output_shapes
:         АТ
EnsureShape_15EnsureShape!UnsortedSegmentMean_6/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АН
"update_connection/ReadVariableOp_3ReadVariableOp)update_connection_readvariableop_resource*
_output_shapes
:	А*
dtype0Й
update_connection/unstack_3Unpack*update_connection/ReadVariableOp_3:value:0*
T0*"
_output_shapes
:А:А*	
numЬ
)update_connection/MatMul_6/ReadVariableOpReadVariableOp0update_connection_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0г
update_connection/MatMul_6MatMulEnsureShape_15:output:01update_connection/MatMul_6/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ае
update_connection/BiasAdd_6BiasAdd$update_connection/MatMul_6:product:0$update_connection/unstack_3:output:0*
T0*(
_output_shapes
:         Аn
#update_connection/split_6/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         т
update_connection/split_6Split,update_connection/split_6/split_dim:output:0$update_connection/BiasAdd_6:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЮ
)update_connection/MatMul_7/ReadVariableOpReadVariableOp2update_connection_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0и
update_connection/MatMul_7MatMulupdate_connection/add_11:z:01update_connection/MatMul_7/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ае
update_connection/BiasAdd_7BiasAdd$update_connection/MatMul_7:product:0$update_connection/unstack_3:output:1*
T0*(
_output_shapes
:         Аn
update_connection/Const_3Const*
_output_shapes
:*
dtype0*!
valueB"А   А       n
#update_connection/split_7/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         У
update_connection/split_7SplitV$update_connection/BiasAdd_7:output:0"update_connection/Const_3:output:0,update_connection/split_7/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЬ
update_connection/add_12AddV2"update_connection/split_6:output:0"update_connection/split_7:output:0*
T0*(
_output_shapes
:         Аw
update_connection/Sigmoid_6Sigmoidupdate_connection/add_12:z:0*
T0*(
_output_shapes
:         АЬ
update_connection/add_13AddV2"update_connection/split_6:output:1"update_connection/split_7:output:1*
T0*(
_output_shapes
:         Аw
update_connection/Sigmoid_7Sigmoidupdate_connection/add_13:z:0*
T0*(
_output_shapes
:         АЦ
update_connection/mul_9Mulupdate_connection/Sigmoid_7:y:0"update_connection/split_7:output:2*
T0*(
_output_shapes
:         АХ
update_connection/add_14AddV2"update_connection/split_6:output:2update_connection/mul_9:z:0*
T0*(
_output_shapes
:         Аq
update_connection/Tanh_3Tanhupdate_connection/add_14:z:0*
T0*(
_output_shapes
:         АС
update_connection/mul_10Mulupdate_connection/Sigmoid_6:y:0update_connection/add_11:z:0*
T0*(
_output_shapes
:         А^
update_connection/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
update_connection/sub_3Sub"update_connection/sub_3/x:output:0update_connection/Sigmoid_6:y:0*
T0*(
_output_shapes
:         АН
update_connection/mul_11Mulupdate_connection/sub_3:z:0update_connection/Tanh_3:y:0*
T0*(
_output_shapes
:         АР
update_connection/add_15AddV2update_connection/mul_10:z:0update_connection/mul_11:z:0*
T0*(
_output_shapes
:         АR
GatherV2_16/axisConst*
_output_shapes
: *
dtype0*
value	B : в
GatherV2_16GatherV2update_ip/add_15:z:0Squeeze_1:output:0GatherV2_16/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_17/axisConst*
_output_shapes
: *
dtype0*
value	B : к
GatherV2_17GatherV2update_connection/add_15:z:0Squeeze_2:output:0GatherV2_17/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_21SqueezeGatherV2_17:output:0*
T0*
_output_shapes
:N

Squeeze_22SqueezeGatherV2_16:output:0*
T0*
_output_shapes
:O
concat_9/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concat_9ConcatV2Squeeze_22:output:0Squeeze_21:output:0concat_9/axis:output:0*
N*
T0*
_output_shapes
:В
EnsureShape_16EnsureShapeconcat_9:output:0*
T0*(
_output_shapes
:         А*
shape:         Аu
sequential/dropout/Identity_4IdentityEnsureShape_16:output:0*
T0*(
_output_shapes
:         АЪ
(sequential/dense/MatMul_4/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0░
sequential/dense/MatMul_4MatMul&sequential/dropout/Identity_4:output:00sequential/dense/MatMul_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЧ
)sequential/dense/BiasAdd_4/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0░
sequential/dense/BiasAdd_4BiasAdd#sequential/dense/MatMul_4:product:01sequential/dense/BiasAdd_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аw
sequential/dense/Relu_4Relu#sequential/dense/BiasAdd_4:output:0*
T0*(
_output_shapes
:         Аt
UnsortedSegmentMean_8/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_8/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_8/onesFill$UnsortedSegmentMean_8/Shape:output:0)UnsortedSegmentMean_8/ones/Const:output:0*
T0*
_output_shapes
:╩
(UnsortedSegmentMean_8/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_8/ones:output:0Squeeze_2:output:0
inputs_n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
#UnsortedSegmentMean_8/strided_sliceStridedSlice
inputs_n_c2UnsortedSegmentMean_8/strided_slice/stack:output:04UnsortedSegmentMean_8/strided_slice/stack_1:output:04UnsortedSegmentMean_8/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_8/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_8/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_8/subSub#UnsortedSegmentMean_8/Rank:output:0%UnsortedSegmentMean_8/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_8/ones_1/packedPackUnsortedSegmentMean_8/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_8/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_8/ones_1Fill,UnsortedSegmentMean_8/ones_1/packed:output:0+UnsortedSegmentMean_8/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_8/concatConcatV2,UnsortedSegmentMean_8/strided_slice:output:0%UnsortedSegmentMean_8/ones_1:output:0*UnsortedSegmentMean_8/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_8/ReshapeReshape1UnsortedSegmentMean_8/UnsortedSegmentSum:output:0%UnsortedSegmentMean_8/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_8/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_8/MaximumMaximum&UnsortedSegmentMean_8/Reshape:output:0(UnsortedSegmentMean_8/Maximum/y:output:0*
T0*
_output_shapes
:╬
*UnsortedSegmentMean_8/UnsortedSegmentSum_1UnsortedSegmentSum%sequential/dense/Relu_4:activations:0Squeeze_2:output:0
inputs_n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_8/truedivRealDiv3UnsortedSegmentMean_8/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_8/Maximum:z:0*
T0*
_output_shapes
:R
GatherV2_18/axisConst*
_output_shapes
: *
dtype0*
value	B : к
GatherV2_18GatherV2update_connection/add_15:z:0Squeeze_3:output:0GatherV2_18/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_19/axisConst*
_output_shapes
: *
dtype0*
value	B : в
GatherV2_19GatherV2update_ip/add_15:z:0Squeeze_4:output:0GatherV2_19/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_23SqueezeGatherV2_19:output:0*
T0*
_output_shapes
:N

Squeeze_24SqueezeGatherV2_18:output:0*
T0*
_output_shapes
:P
concat_10/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_10ConcatV2Squeeze_24:output:0Squeeze_23:output:0concat_10/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_17EnsureShapeconcat_10:output:0*
T0*(
_output_shapes
:         А*
shape:         Аy
!sequential_1/dropout_1/Identity_4IdentityEnsureShape_17:output:0*
T0*(
_output_shapes
:         Ав
,sequential_1/dense_1/MatMul_4/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╝
sequential_1/dense_1/MatMul_4MatMul*sequential_1/dropout_1/Identity_4:output:04sequential_1/dense_1/MatMul_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
-sequential_1/dense_1/BiasAdd_4/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╝
sequential_1/dense_1/BiasAdd_4BiasAdd'sequential_1/dense_1/MatMul_4:product:05sequential_1/dense_1/BiasAdd_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А
sequential_1/dense_1/Relu_4Relu'sequential_1/dense_1/BiasAdd_4:output:0*
T0*(
_output_shapes
:         Аt
UnsortedSegmentMean_9/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_9/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_9/onesFill$UnsortedSegmentMean_9/Shape:output:0)UnsortedSegmentMean_9/ones/Const:output:0*
T0*
_output_shapes
:╩
(UnsortedSegmentMean_9/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_9/ones:output:0Squeeze_4:output:0
inputs_n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
#UnsortedSegmentMean_9/strided_sliceStridedSlice
inputs_n_i2UnsortedSegmentMean_9/strided_slice/stack:output:04UnsortedSegmentMean_9/strided_slice/stack_1:output:04UnsortedSegmentMean_9/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_9/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_9/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_9/subSub#UnsortedSegmentMean_9/Rank:output:0%UnsortedSegmentMean_9/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_9/ones_1/packedPackUnsortedSegmentMean_9/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_9/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_9/ones_1Fill,UnsortedSegmentMean_9/ones_1/packed:output:0+UnsortedSegmentMean_9/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_9/concatConcatV2,UnsortedSegmentMean_9/strided_slice:output:0%UnsortedSegmentMean_9/ones_1:output:0*UnsortedSegmentMean_9/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_9/ReshapeReshape1UnsortedSegmentMean_9/UnsortedSegmentSum:output:0%UnsortedSegmentMean_9/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_9/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_9/MaximumMaximum&UnsortedSegmentMean_9/Reshape:output:0(UnsortedSegmentMean_9/Maximum/y:output:0*
T0*
_output_shapes
:╥
*UnsortedSegmentMean_9/UnsortedSegmentSum_1UnsortedSegmentSum)sequential_1/dense_1/Relu_4:activations:0Squeeze_4:output:0
inputs_n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_9/truedivRealDiv3UnsortedSegmentMean_9/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_9/Maximum:z:0*
T0*
_output_shapes
:Т
EnsureShape_18EnsureShape!UnsortedSegmentMean_9/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         А}
update_ip/ReadVariableOp_4ReadVariableOp!update_ip_readvariableop_resource*
_output_shapes
:	А*
dtype0y
update_ip/unstack_4Unpack"update_ip/ReadVariableOp_4:value:0*
T0*"
_output_shapes
:А:А*	
numМ
!update_ip/MatMul_8/ReadVariableOpReadVariableOp(update_ip_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0У
update_ip/MatMul_8MatMulEnsureShape_18:output:0)update_ip/MatMul_8/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АН
update_ip/BiasAdd_8BiasAddupdate_ip/MatMul_8:product:0update_ip/unstack_4:output:0*
T0*(
_output_shapes
:         Аf
update_ip/split_8/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╩
update_ip/split_8Split$update_ip/split_8/split_dim:output:0update_ip/BiasAdd_8:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitО
!update_ip/MatMul_9/ReadVariableOpReadVariableOp*update_ip_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Р
update_ip/MatMul_9MatMulupdate_ip/add_15:z:0)update_ip/MatMul_9/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АН
update_ip/BiasAdd_9BiasAddupdate_ip/MatMul_9:product:0update_ip/unstack_4:output:1*
T0*(
_output_shapes
:         Аf
update_ip/Const_4Const*
_output_shapes
:*
dtype0*!
valueB"А   А       f
update_ip/split_9/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         є
update_ip/split_9SplitVupdate_ip/BiasAdd_9:output:0update_ip/Const_4:output:0$update_ip/split_9/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitД
update_ip/add_16AddV2update_ip/split_8:output:0update_ip/split_9:output:0*
T0*(
_output_shapes
:         Аg
update_ip/Sigmoid_8Sigmoidupdate_ip/add_16:z:0*
T0*(
_output_shapes
:         АД
update_ip/add_17AddV2update_ip/split_8:output:1update_ip/split_9:output:1*
T0*(
_output_shapes
:         Аg
update_ip/Sigmoid_9Sigmoidupdate_ip/add_17:z:0*
T0*(
_output_shapes
:         А
update_ip/mul_12Mulupdate_ip/Sigmoid_9:y:0update_ip/split_9:output:2*
T0*(
_output_shapes
:         А~
update_ip/add_18AddV2update_ip/split_8:output:2update_ip/mul_12:z:0*
T0*(
_output_shapes
:         Аa
update_ip/Tanh_4Tanhupdate_ip/add_18:z:0*
T0*(
_output_shapes
:         Аy
update_ip/mul_13Mulupdate_ip/Sigmoid_8:y:0update_ip/add_15:z:0*
T0*(
_output_shapes
:         АV
update_ip/sub_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?~
update_ip/sub_4Subupdate_ip/sub_4/x:output:0update_ip/Sigmoid_8:y:0*
T0*(
_output_shapes
:         Аu
update_ip/mul_14Mulupdate_ip/sub_4:z:0update_ip/Tanh_4:y:0*
T0*(
_output_shapes
:         Аx
update_ip/add_19AddV2update_ip/mul_13:z:0update_ip/mul_14:z:0*
T0*(
_output_shapes
:         АТ
EnsureShape_19EnsureShape!UnsortedSegmentMean_8/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АН
"update_connection/ReadVariableOp_4ReadVariableOp)update_connection_readvariableop_resource*
_output_shapes
:	А*
dtype0Й
update_connection/unstack_4Unpack*update_connection/ReadVariableOp_4:value:0*
T0*"
_output_shapes
:А:А*	
numЬ
)update_connection/MatMul_8/ReadVariableOpReadVariableOp0update_connection_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0г
update_connection/MatMul_8MatMulEnsureShape_19:output:01update_connection/MatMul_8/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ае
update_connection/BiasAdd_8BiasAdd$update_connection/MatMul_8:product:0$update_connection/unstack_4:output:0*
T0*(
_output_shapes
:         Аn
#update_connection/split_8/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         т
update_connection/split_8Split,update_connection/split_8/split_dim:output:0$update_connection/BiasAdd_8:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЮ
)update_connection/MatMul_9/ReadVariableOpReadVariableOp2update_connection_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0и
update_connection/MatMul_9MatMulupdate_connection/add_15:z:01update_connection/MatMul_9/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ае
update_connection/BiasAdd_9BiasAdd$update_connection/MatMul_9:product:0$update_connection/unstack_4:output:1*
T0*(
_output_shapes
:         Аn
update_connection/Const_4Const*
_output_shapes
:*
dtype0*!
valueB"А   А       n
#update_connection/split_9/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         У
update_connection/split_9SplitV$update_connection/BiasAdd_9:output:0"update_connection/Const_4:output:0,update_connection/split_9/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЬ
update_connection/add_16AddV2"update_connection/split_8:output:0"update_connection/split_9:output:0*
T0*(
_output_shapes
:         Аw
update_connection/Sigmoid_8Sigmoidupdate_connection/add_16:z:0*
T0*(
_output_shapes
:         АЬ
update_connection/add_17AddV2"update_connection/split_8:output:1"update_connection/split_9:output:1*
T0*(
_output_shapes
:         Аw
update_connection/Sigmoid_9Sigmoidupdate_connection/add_17:z:0*
T0*(
_output_shapes
:         АЧ
update_connection/mul_12Mulupdate_connection/Sigmoid_9:y:0"update_connection/split_9:output:2*
T0*(
_output_shapes
:         АЦ
update_connection/add_18AddV2"update_connection/split_8:output:2update_connection/mul_12:z:0*
T0*(
_output_shapes
:         Аq
update_connection/Tanh_4Tanhupdate_connection/add_18:z:0*
T0*(
_output_shapes
:         АС
update_connection/mul_13Mulupdate_connection/Sigmoid_8:y:0update_connection/add_15:z:0*
T0*(
_output_shapes
:         А^
update_connection/sub_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
update_connection/sub_4Sub"update_connection/sub_4/x:output:0update_connection/Sigmoid_8:y:0*
T0*(
_output_shapes
:         АН
update_connection/mul_14Mulupdate_connection/sub_4:z:0update_connection/Tanh_4:y:0*
T0*(
_output_shapes
:         АР
update_connection/add_19AddV2update_connection/mul_13:z:0update_connection/mul_14:z:0*
T0*(
_output_shapes
:         АR
GatherV2_20/axisConst*
_output_shapes
: *
dtype0*
value	B : в
GatherV2_20GatherV2update_ip/add_19:z:0Squeeze_1:output:0GatherV2_20/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_21/axisConst*
_output_shapes
: *
dtype0*
value	B : к
GatherV2_21GatherV2update_connection/add_19:z:0Squeeze_2:output:0GatherV2_21/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_25SqueezeGatherV2_21:output:0*
T0*
_output_shapes
:N

Squeeze_26SqueezeGatherV2_20:output:0*
T0*
_output_shapes
:P
concat_11/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_11ConcatV2Squeeze_26:output:0Squeeze_25:output:0concat_11/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_20EnsureShapeconcat_11:output:0*
T0*(
_output_shapes
:         А*
shape:         Аu
sequential/dropout/Identity_5IdentityEnsureShape_20:output:0*
T0*(
_output_shapes
:         АЪ
(sequential/dense/MatMul_5/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0░
sequential/dense/MatMul_5MatMul&sequential/dropout/Identity_5:output:00sequential/dense/MatMul_5/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЧ
)sequential/dense/BiasAdd_5/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0░
sequential/dense/BiasAdd_5BiasAdd#sequential/dense/MatMul_5:product:01sequential/dense/BiasAdd_5/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аw
sequential/dense/Relu_5Relu#sequential/dense/BiasAdd_5:output:0*
T0*(
_output_shapes
:         Аu
UnsortedSegmentMean_10/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧f
!UnsortedSegmentMean_10/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
UnsortedSegmentMean_10/onesFill%UnsortedSegmentMean_10/Shape:output:0*UnsortedSegmentMean_10/ones/Const:output:0*
T0*
_output_shapes
:╠
)UnsortedSegmentMean_10/UnsortedSegmentSumUnsortedSegmentSum$UnsortedSegmentMean_10/ones:output:0Squeeze_2:output:0
inputs_n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:t
*UnsortedSegmentMean_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
$UnsortedSegmentMean_10/strided_sliceStridedSlice
inputs_n_c3UnsortedSegmentMean_10/strided_slice/stack:output:05UnsortedSegmentMean_10/strided_slice/stack_1:output:05UnsortedSegmentMean_10/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask]
UnsortedSegmentMean_10/RankConst*
_output_shapes
: *
dtype0*
value	B :Z
UnsortedSegmentMean_10/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Р
UnsortedSegmentMean_10/subSub$UnsortedSegmentMean_10/Rank:output:0&UnsortedSegmentMean_10/Rank_1:output:0*
T0*
_output_shapes
: z
$UnsortedSegmentMean_10/ones_1/packedPackUnsortedSegmentMean_10/sub:z:0*
N*
T0*
_output_shapes
:e
#UnsortedSegmentMean_10/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R░
UnsortedSegmentMean_10/ones_1Fill-UnsortedSegmentMean_10/ones_1/packed:output:0,UnsortedSegmentMean_10/ones_1/Const:output:0*
T0	*#
_output_shapes
:         d
"UnsortedSegmentMean_10/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
UnsortedSegmentMean_10/concatConcatV2-UnsortedSegmentMean_10/strided_slice:output:0&UnsortedSegmentMean_10/ones_1:output:0+UnsortedSegmentMean_10/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         ╢
UnsortedSegmentMean_10/ReshapeReshape2UnsortedSegmentMean_10/UnsortedSegmentSum:output:0&UnsortedSegmentMean_10/concat:output:0*
Tshape0	*
T0*
_output_shapes
:e
 UnsortedSegmentMean_10/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
UnsortedSegmentMean_10/MaximumMaximum'UnsortedSegmentMean_10/Reshape:output:0)UnsortedSegmentMean_10/Maximum/y:output:0*
T0*
_output_shapes
:╧
+UnsortedSegmentMean_10/UnsortedSegmentSum_1UnsortedSegmentSum%sequential/dense/Relu_5:activations:0Squeeze_2:output:0
inputs_n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:ж
UnsortedSegmentMean_10/truedivRealDiv4UnsortedSegmentMean_10/UnsortedSegmentSum_1:output:0"UnsortedSegmentMean_10/Maximum:z:0*
T0*
_output_shapes
:R
GatherV2_22/axisConst*
_output_shapes
: *
dtype0*
value	B : к
GatherV2_22GatherV2update_connection/add_19:z:0Squeeze_3:output:0GatherV2_22/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_23/axisConst*
_output_shapes
: *
dtype0*
value	B : в
GatherV2_23GatherV2update_ip/add_19:z:0Squeeze_4:output:0GatherV2_23/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_27SqueezeGatherV2_23:output:0*
T0*
_output_shapes
:N

Squeeze_28SqueezeGatherV2_22:output:0*
T0*
_output_shapes
:P
concat_12/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_12ConcatV2Squeeze_28:output:0Squeeze_27:output:0concat_12/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_21EnsureShapeconcat_12:output:0*
T0*(
_output_shapes
:         А*
shape:         Аy
!sequential_1/dropout_1/Identity_5IdentityEnsureShape_21:output:0*
T0*(
_output_shapes
:         Ав
,sequential_1/dense_1/MatMul_5/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╝
sequential_1/dense_1/MatMul_5MatMul*sequential_1/dropout_1/Identity_5:output:04sequential_1/dense_1/MatMul_5/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
-sequential_1/dense_1/BiasAdd_5/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╝
sequential_1/dense_1/BiasAdd_5BiasAdd'sequential_1/dense_1/MatMul_5:product:05sequential_1/dense_1/BiasAdd_5/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А
sequential_1/dense_1/Relu_5Relu'sequential_1/dense_1/BiasAdd_5:output:0*
T0*(
_output_shapes
:         Аu
UnsortedSegmentMean_11/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧f
!UnsortedSegmentMean_11/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
UnsortedSegmentMean_11/onesFill%UnsortedSegmentMean_11/Shape:output:0*UnsortedSegmentMean_11/ones/Const:output:0*
T0*
_output_shapes
:╠
)UnsortedSegmentMean_11/UnsortedSegmentSumUnsortedSegmentSum$UnsortedSegmentMean_11/ones:output:0Squeeze_4:output:0
inputs_n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:t
*UnsortedSegmentMean_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
$UnsortedSegmentMean_11/strided_sliceStridedSlice
inputs_n_i3UnsortedSegmentMean_11/strided_slice/stack:output:05UnsortedSegmentMean_11/strided_slice/stack_1:output:05UnsortedSegmentMean_11/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask]
UnsortedSegmentMean_11/RankConst*
_output_shapes
: *
dtype0*
value	B :Z
UnsortedSegmentMean_11/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Р
UnsortedSegmentMean_11/subSub$UnsortedSegmentMean_11/Rank:output:0&UnsortedSegmentMean_11/Rank_1:output:0*
T0*
_output_shapes
: z
$UnsortedSegmentMean_11/ones_1/packedPackUnsortedSegmentMean_11/sub:z:0*
N*
T0*
_output_shapes
:e
#UnsortedSegmentMean_11/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R░
UnsortedSegmentMean_11/ones_1Fill-UnsortedSegmentMean_11/ones_1/packed:output:0,UnsortedSegmentMean_11/ones_1/Const:output:0*
T0	*#
_output_shapes
:         d
"UnsortedSegmentMean_11/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
UnsortedSegmentMean_11/concatConcatV2-UnsortedSegmentMean_11/strided_slice:output:0&UnsortedSegmentMean_11/ones_1:output:0+UnsortedSegmentMean_11/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         ╢
UnsortedSegmentMean_11/ReshapeReshape2UnsortedSegmentMean_11/UnsortedSegmentSum:output:0&UnsortedSegmentMean_11/concat:output:0*
Tshape0	*
T0*
_output_shapes
:e
 UnsortedSegmentMean_11/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
UnsortedSegmentMean_11/MaximumMaximum'UnsortedSegmentMean_11/Reshape:output:0)UnsortedSegmentMean_11/Maximum/y:output:0*
T0*
_output_shapes
:╙
+UnsortedSegmentMean_11/UnsortedSegmentSum_1UnsortedSegmentSum)sequential_1/dense_1/Relu_5:activations:0Squeeze_4:output:0
inputs_n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:ж
UnsortedSegmentMean_11/truedivRealDiv4UnsortedSegmentMean_11/UnsortedSegmentSum_1:output:0"UnsortedSegmentMean_11/Maximum:z:0*
T0*
_output_shapes
:У
EnsureShape_22EnsureShape"UnsortedSegmentMean_11/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         А}
update_ip/ReadVariableOp_5ReadVariableOp!update_ip_readvariableop_resource*
_output_shapes
:	А*
dtype0y
update_ip/unstack_5Unpack"update_ip/ReadVariableOp_5:value:0*
T0*"
_output_shapes
:А:А*	
numН
"update_ip/MatMul_10/ReadVariableOpReadVariableOp(update_ip_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Х
update_ip/MatMul_10MatMulEnsureShape_22:output:0*update_ip/MatMul_10/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АП
update_ip/BiasAdd_10BiasAddupdate_ip/MatMul_10:product:0update_ip/unstack_5:output:0*
T0*(
_output_shapes
:         Аg
update_ip/split_10/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ═
update_ip/split_10Split%update_ip/split_10/split_dim:output:0update_ip/BiasAdd_10:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitП
"update_ip/MatMul_11/ReadVariableOpReadVariableOp*update_ip_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Т
update_ip/MatMul_11MatMulupdate_ip/add_19:z:0*update_ip/MatMul_11/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АП
update_ip/BiasAdd_11BiasAddupdate_ip/MatMul_11:product:0update_ip/unstack_5:output:1*
T0*(
_output_shapes
:         Аf
update_ip/Const_5Const*
_output_shapes
:*
dtype0*!
valueB"А   А       g
update_ip/split_11/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Ў
update_ip/split_11SplitVupdate_ip/BiasAdd_11:output:0update_ip/Const_5:output:0%update_ip/split_11/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЖ
update_ip/add_20AddV2update_ip/split_10:output:0update_ip/split_11:output:0*
T0*(
_output_shapes
:         Аh
update_ip/Sigmoid_10Sigmoidupdate_ip/add_20:z:0*
T0*(
_output_shapes
:         АЖ
update_ip/add_21AddV2update_ip/split_10:output:1update_ip/split_11:output:1*
T0*(
_output_shapes
:         Аh
update_ip/Sigmoid_11Sigmoidupdate_ip/add_21:z:0*
T0*(
_output_shapes
:         АБ
update_ip/mul_15Mulupdate_ip/Sigmoid_11:y:0update_ip/split_11:output:2*
T0*(
_output_shapes
:         А
update_ip/add_22AddV2update_ip/split_10:output:2update_ip/mul_15:z:0*
T0*(
_output_shapes
:         Аa
update_ip/Tanh_5Tanhupdate_ip/add_22:z:0*
T0*(
_output_shapes
:         Аz
update_ip/mul_16Mulupdate_ip/Sigmoid_10:y:0update_ip/add_19:z:0*
T0*(
_output_shapes
:         АV
update_ip/sub_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
update_ip/sub_5Subupdate_ip/sub_5/x:output:0update_ip/Sigmoid_10:y:0*
T0*(
_output_shapes
:         Аu
update_ip/mul_17Mulupdate_ip/sub_5:z:0update_ip/Tanh_5:y:0*
T0*(
_output_shapes
:         Аx
update_ip/add_23AddV2update_ip/mul_16:z:0update_ip/mul_17:z:0*
T0*(
_output_shapes
:         АУ
EnsureShape_23EnsureShape"UnsortedSegmentMean_10/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АН
"update_connection/ReadVariableOp_5ReadVariableOp)update_connection_readvariableop_resource*
_output_shapes
:	А*
dtype0Й
update_connection/unstack_5Unpack*update_connection/ReadVariableOp_5:value:0*
T0*"
_output_shapes
:А:А*	
numЭ
*update_connection/MatMul_10/ReadVariableOpReadVariableOp0update_connection_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0е
update_connection/MatMul_10MatMulEnsureShape_23:output:02update_connection/MatMul_10/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аз
update_connection/BiasAdd_10BiasAdd%update_connection/MatMul_10:product:0$update_connection/unstack_5:output:0*
T0*(
_output_shapes
:         Аo
$update_connection/split_10/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         х
update_connection/split_10Split-update_connection/split_10/split_dim:output:0%update_connection/BiasAdd_10:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЯ
*update_connection/MatMul_11/ReadVariableOpReadVariableOp2update_connection_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0к
update_connection/MatMul_11MatMulupdate_connection/add_19:z:02update_connection/MatMul_11/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аз
update_connection/BiasAdd_11BiasAdd%update_connection/MatMul_11:product:0$update_connection/unstack_5:output:1*
T0*(
_output_shapes
:         Аn
update_connection/Const_5Const*
_output_shapes
:*
dtype0*!
valueB"А   А       o
$update_connection/split_11/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Ц
update_connection/split_11SplitV%update_connection/BiasAdd_11:output:0"update_connection/Const_5:output:0-update_connection/split_11/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЮ
update_connection/add_20AddV2#update_connection/split_10:output:0#update_connection/split_11:output:0*
T0*(
_output_shapes
:         Аx
update_connection/Sigmoid_10Sigmoidupdate_connection/add_20:z:0*
T0*(
_output_shapes
:         АЮ
update_connection/add_21AddV2#update_connection/split_10:output:1#update_connection/split_11:output:1*
T0*(
_output_shapes
:         Аx
update_connection/Sigmoid_11Sigmoidupdate_connection/add_21:z:0*
T0*(
_output_shapes
:         АЩ
update_connection/mul_15Mul update_connection/Sigmoid_11:y:0#update_connection/split_11:output:2*
T0*(
_output_shapes
:         АЧ
update_connection/add_22AddV2#update_connection/split_10:output:2update_connection/mul_15:z:0*
T0*(
_output_shapes
:         Аq
update_connection/Tanh_5Tanhupdate_connection/add_22:z:0*
T0*(
_output_shapes
:         АТ
update_connection/mul_16Mul update_connection/Sigmoid_10:y:0update_connection/add_19:z:0*
T0*(
_output_shapes
:         А^
update_connection/sub_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ч
update_connection/sub_5Sub"update_connection/sub_5/x:output:0 update_connection/Sigmoid_10:y:0*
T0*(
_output_shapes
:         АН
update_connection/mul_17Mulupdate_connection/sub_5:z:0update_connection/Tanh_5:y:0*
T0*(
_output_shapes
:         АР
update_connection/add_23AddV2update_connection/mul_16:z:0update_connection/mul_17:z:0*
T0*(
_output_shapes
:         АR
GatherV2_24/axisConst*
_output_shapes
: *
dtype0*
value	B : в
GatherV2_24GatherV2update_ip/add_23:z:0Squeeze_1:output:0GatherV2_24/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_25/axisConst*
_output_shapes
: *
dtype0*
value	B : к
GatherV2_25GatherV2update_connection/add_23:z:0Squeeze_2:output:0GatherV2_25/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_29SqueezeGatherV2_25:output:0*
T0*
_output_shapes
:N

Squeeze_30SqueezeGatherV2_24:output:0*
T0*
_output_shapes
:P
concat_13/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_13ConcatV2Squeeze_30:output:0Squeeze_29:output:0concat_13/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_24EnsureShapeconcat_13:output:0*
T0*(
_output_shapes
:         А*
shape:         Аu
sequential/dropout/Identity_6IdentityEnsureShape_24:output:0*
T0*(
_output_shapes
:         АЪ
(sequential/dense/MatMul_6/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0░
sequential/dense/MatMul_6MatMul&sequential/dropout/Identity_6:output:00sequential/dense/MatMul_6/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЧ
)sequential/dense/BiasAdd_6/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0░
sequential/dense/BiasAdd_6BiasAdd#sequential/dense/MatMul_6:product:01sequential/dense/BiasAdd_6/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аw
sequential/dense/Relu_6Relu#sequential/dense/BiasAdd_6:output:0*
T0*(
_output_shapes
:         Аu
UnsortedSegmentMean_12/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧f
!UnsortedSegmentMean_12/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
UnsortedSegmentMean_12/onesFill%UnsortedSegmentMean_12/Shape:output:0*UnsortedSegmentMean_12/ones/Const:output:0*
T0*
_output_shapes
:╠
)UnsortedSegmentMean_12/UnsortedSegmentSumUnsortedSegmentSum$UnsortedSegmentMean_12/ones:output:0Squeeze_2:output:0
inputs_n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:t
*UnsortedSegmentMean_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
$UnsortedSegmentMean_12/strided_sliceStridedSlice
inputs_n_c3UnsortedSegmentMean_12/strided_slice/stack:output:05UnsortedSegmentMean_12/strided_slice/stack_1:output:05UnsortedSegmentMean_12/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask]
UnsortedSegmentMean_12/RankConst*
_output_shapes
: *
dtype0*
value	B :Z
UnsortedSegmentMean_12/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Р
UnsortedSegmentMean_12/subSub$UnsortedSegmentMean_12/Rank:output:0&UnsortedSegmentMean_12/Rank_1:output:0*
T0*
_output_shapes
: z
$UnsortedSegmentMean_12/ones_1/packedPackUnsortedSegmentMean_12/sub:z:0*
N*
T0*
_output_shapes
:e
#UnsortedSegmentMean_12/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R░
UnsortedSegmentMean_12/ones_1Fill-UnsortedSegmentMean_12/ones_1/packed:output:0,UnsortedSegmentMean_12/ones_1/Const:output:0*
T0	*#
_output_shapes
:         d
"UnsortedSegmentMean_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
UnsortedSegmentMean_12/concatConcatV2-UnsortedSegmentMean_12/strided_slice:output:0&UnsortedSegmentMean_12/ones_1:output:0+UnsortedSegmentMean_12/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         ╢
UnsortedSegmentMean_12/ReshapeReshape2UnsortedSegmentMean_12/UnsortedSegmentSum:output:0&UnsortedSegmentMean_12/concat:output:0*
Tshape0	*
T0*
_output_shapes
:e
 UnsortedSegmentMean_12/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
UnsortedSegmentMean_12/MaximumMaximum'UnsortedSegmentMean_12/Reshape:output:0)UnsortedSegmentMean_12/Maximum/y:output:0*
T0*
_output_shapes
:╧
+UnsortedSegmentMean_12/UnsortedSegmentSum_1UnsortedSegmentSum%sequential/dense/Relu_6:activations:0Squeeze_2:output:0
inputs_n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:ж
UnsortedSegmentMean_12/truedivRealDiv4UnsortedSegmentMean_12/UnsortedSegmentSum_1:output:0"UnsortedSegmentMean_12/Maximum:z:0*
T0*
_output_shapes
:R
GatherV2_26/axisConst*
_output_shapes
: *
dtype0*
value	B : к
GatherV2_26GatherV2update_connection/add_23:z:0Squeeze_3:output:0GatherV2_26/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_27/axisConst*
_output_shapes
: *
dtype0*
value	B : в
GatherV2_27GatherV2update_ip/add_23:z:0Squeeze_4:output:0GatherV2_27/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_31SqueezeGatherV2_27:output:0*
T0*
_output_shapes
:N

Squeeze_32SqueezeGatherV2_26:output:0*
T0*
_output_shapes
:P
concat_14/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_14ConcatV2Squeeze_32:output:0Squeeze_31:output:0concat_14/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_25EnsureShapeconcat_14:output:0*
T0*(
_output_shapes
:         А*
shape:         Аy
!sequential_1/dropout_1/Identity_6IdentityEnsureShape_25:output:0*
T0*(
_output_shapes
:         Ав
,sequential_1/dense_1/MatMul_6/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╝
sequential_1/dense_1/MatMul_6MatMul*sequential_1/dropout_1/Identity_6:output:04sequential_1/dense_1/MatMul_6/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
-sequential_1/dense_1/BiasAdd_6/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╝
sequential_1/dense_1/BiasAdd_6BiasAdd'sequential_1/dense_1/MatMul_6:product:05sequential_1/dense_1/BiasAdd_6/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А
sequential_1/dense_1/Relu_6Relu'sequential_1/dense_1/BiasAdd_6:output:0*
T0*(
_output_shapes
:         Аu
UnsortedSegmentMean_13/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧f
!UnsortedSegmentMean_13/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
UnsortedSegmentMean_13/onesFill%UnsortedSegmentMean_13/Shape:output:0*UnsortedSegmentMean_13/ones/Const:output:0*
T0*
_output_shapes
:╠
)UnsortedSegmentMean_13/UnsortedSegmentSumUnsortedSegmentSum$UnsortedSegmentMean_13/ones:output:0Squeeze_4:output:0
inputs_n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:t
*UnsortedSegmentMean_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
$UnsortedSegmentMean_13/strided_sliceStridedSlice
inputs_n_i3UnsortedSegmentMean_13/strided_slice/stack:output:05UnsortedSegmentMean_13/strided_slice/stack_1:output:05UnsortedSegmentMean_13/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask]
UnsortedSegmentMean_13/RankConst*
_output_shapes
: *
dtype0*
value	B :Z
UnsortedSegmentMean_13/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Р
UnsortedSegmentMean_13/subSub$UnsortedSegmentMean_13/Rank:output:0&UnsortedSegmentMean_13/Rank_1:output:0*
T0*
_output_shapes
: z
$UnsortedSegmentMean_13/ones_1/packedPackUnsortedSegmentMean_13/sub:z:0*
N*
T0*
_output_shapes
:e
#UnsortedSegmentMean_13/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R░
UnsortedSegmentMean_13/ones_1Fill-UnsortedSegmentMean_13/ones_1/packed:output:0,UnsortedSegmentMean_13/ones_1/Const:output:0*
T0	*#
_output_shapes
:         d
"UnsortedSegmentMean_13/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
UnsortedSegmentMean_13/concatConcatV2-UnsortedSegmentMean_13/strided_slice:output:0&UnsortedSegmentMean_13/ones_1:output:0+UnsortedSegmentMean_13/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         ╢
UnsortedSegmentMean_13/ReshapeReshape2UnsortedSegmentMean_13/UnsortedSegmentSum:output:0&UnsortedSegmentMean_13/concat:output:0*
Tshape0	*
T0*
_output_shapes
:e
 UnsortedSegmentMean_13/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
UnsortedSegmentMean_13/MaximumMaximum'UnsortedSegmentMean_13/Reshape:output:0)UnsortedSegmentMean_13/Maximum/y:output:0*
T0*
_output_shapes
:╙
+UnsortedSegmentMean_13/UnsortedSegmentSum_1UnsortedSegmentSum)sequential_1/dense_1/Relu_6:activations:0Squeeze_4:output:0
inputs_n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:ж
UnsortedSegmentMean_13/truedivRealDiv4UnsortedSegmentMean_13/UnsortedSegmentSum_1:output:0"UnsortedSegmentMean_13/Maximum:z:0*
T0*
_output_shapes
:У
EnsureShape_26EnsureShape"UnsortedSegmentMean_13/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         А}
update_ip/ReadVariableOp_6ReadVariableOp!update_ip_readvariableop_resource*
_output_shapes
:	А*
dtype0y
update_ip/unstack_6Unpack"update_ip/ReadVariableOp_6:value:0*
T0*"
_output_shapes
:А:А*	
numН
"update_ip/MatMul_12/ReadVariableOpReadVariableOp(update_ip_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Х
update_ip/MatMul_12MatMulEnsureShape_26:output:0*update_ip/MatMul_12/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АП
update_ip/BiasAdd_12BiasAddupdate_ip/MatMul_12:product:0update_ip/unstack_6:output:0*
T0*(
_output_shapes
:         Аg
update_ip/split_12/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ═
update_ip/split_12Split%update_ip/split_12/split_dim:output:0update_ip/BiasAdd_12:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitП
"update_ip/MatMul_13/ReadVariableOpReadVariableOp*update_ip_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Т
update_ip/MatMul_13MatMulupdate_ip/add_23:z:0*update_ip/MatMul_13/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АП
update_ip/BiasAdd_13BiasAddupdate_ip/MatMul_13:product:0update_ip/unstack_6:output:1*
T0*(
_output_shapes
:         Аf
update_ip/Const_6Const*
_output_shapes
:*
dtype0*!
valueB"А   А       g
update_ip/split_13/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Ў
update_ip/split_13SplitVupdate_ip/BiasAdd_13:output:0update_ip/Const_6:output:0%update_ip/split_13/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЖ
update_ip/add_24AddV2update_ip/split_12:output:0update_ip/split_13:output:0*
T0*(
_output_shapes
:         Аh
update_ip/Sigmoid_12Sigmoidupdate_ip/add_24:z:0*
T0*(
_output_shapes
:         АЖ
update_ip/add_25AddV2update_ip/split_12:output:1update_ip/split_13:output:1*
T0*(
_output_shapes
:         Аh
update_ip/Sigmoid_13Sigmoidupdate_ip/add_25:z:0*
T0*(
_output_shapes
:         АБ
update_ip/mul_18Mulupdate_ip/Sigmoid_13:y:0update_ip/split_13:output:2*
T0*(
_output_shapes
:         А
update_ip/add_26AddV2update_ip/split_12:output:2update_ip/mul_18:z:0*
T0*(
_output_shapes
:         Аa
update_ip/Tanh_6Tanhupdate_ip/add_26:z:0*
T0*(
_output_shapes
:         Аz
update_ip/mul_19Mulupdate_ip/Sigmoid_12:y:0update_ip/add_23:z:0*
T0*(
_output_shapes
:         АV
update_ip/sub_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
update_ip/sub_6Subupdate_ip/sub_6/x:output:0update_ip/Sigmoid_12:y:0*
T0*(
_output_shapes
:         Аu
update_ip/mul_20Mulupdate_ip/sub_6:z:0update_ip/Tanh_6:y:0*
T0*(
_output_shapes
:         Аx
update_ip/add_27AddV2update_ip/mul_19:z:0update_ip/mul_20:z:0*
T0*(
_output_shapes
:         АУ
EnsureShape_27EnsureShape"UnsortedSegmentMean_12/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АН
"update_connection/ReadVariableOp_6ReadVariableOp)update_connection_readvariableop_resource*
_output_shapes
:	А*
dtype0Й
update_connection/unstack_6Unpack*update_connection/ReadVariableOp_6:value:0*
T0*"
_output_shapes
:А:А*	
numЭ
*update_connection/MatMul_12/ReadVariableOpReadVariableOp0update_connection_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0е
update_connection/MatMul_12MatMulEnsureShape_27:output:02update_connection/MatMul_12/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аз
update_connection/BiasAdd_12BiasAdd%update_connection/MatMul_12:product:0$update_connection/unstack_6:output:0*
T0*(
_output_shapes
:         Аo
$update_connection/split_12/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         х
update_connection/split_12Split-update_connection/split_12/split_dim:output:0%update_connection/BiasAdd_12:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЯ
*update_connection/MatMul_13/ReadVariableOpReadVariableOp2update_connection_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0к
update_connection/MatMul_13MatMulupdate_connection/add_23:z:02update_connection/MatMul_13/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аз
update_connection/BiasAdd_13BiasAdd%update_connection/MatMul_13:product:0$update_connection/unstack_6:output:1*
T0*(
_output_shapes
:         Аn
update_connection/Const_6Const*
_output_shapes
:*
dtype0*!
valueB"А   А       o
$update_connection/split_13/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Ц
update_connection/split_13SplitV%update_connection/BiasAdd_13:output:0"update_connection/Const_6:output:0-update_connection/split_13/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЮ
update_connection/add_24AddV2#update_connection/split_12:output:0#update_connection/split_13:output:0*
T0*(
_output_shapes
:         Аx
update_connection/Sigmoid_12Sigmoidupdate_connection/add_24:z:0*
T0*(
_output_shapes
:         АЮ
update_connection/add_25AddV2#update_connection/split_12:output:1#update_connection/split_13:output:1*
T0*(
_output_shapes
:         Аx
update_connection/Sigmoid_13Sigmoidupdate_connection/add_25:z:0*
T0*(
_output_shapes
:         АЩ
update_connection/mul_18Mul update_connection/Sigmoid_13:y:0#update_connection/split_13:output:2*
T0*(
_output_shapes
:         АЧ
update_connection/add_26AddV2#update_connection/split_12:output:2update_connection/mul_18:z:0*
T0*(
_output_shapes
:         Аq
update_connection/Tanh_6Tanhupdate_connection/add_26:z:0*
T0*(
_output_shapes
:         АТ
update_connection/mul_19Mul update_connection/Sigmoid_12:y:0update_connection/add_23:z:0*
T0*(
_output_shapes
:         А^
update_connection/sub_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ч
update_connection/sub_6Sub"update_connection/sub_6/x:output:0 update_connection/Sigmoid_12:y:0*
T0*(
_output_shapes
:         АН
update_connection/mul_20Mulupdate_connection/sub_6:z:0update_connection/Tanh_6:y:0*
T0*(
_output_shapes
:         АР
update_connection/add_27AddV2update_connection/mul_19:z:0update_connection/mul_20:z:0*
T0*(
_output_shapes
:         АR
GatherV2_28/axisConst*
_output_shapes
: *
dtype0*
value	B : в
GatherV2_28GatherV2update_ip/add_27:z:0Squeeze_1:output:0GatherV2_28/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_29/axisConst*
_output_shapes
: *
dtype0*
value	B : к
GatherV2_29GatherV2update_connection/add_27:z:0Squeeze_2:output:0GatherV2_29/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_33SqueezeGatherV2_29:output:0*
T0*
_output_shapes
:N

Squeeze_34SqueezeGatherV2_28:output:0*
T0*
_output_shapes
:P
concat_15/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_15ConcatV2Squeeze_34:output:0Squeeze_33:output:0concat_15/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_28EnsureShapeconcat_15:output:0*
T0*(
_output_shapes
:         А*
shape:         Аu
sequential/dropout/Identity_7IdentityEnsureShape_28:output:0*
T0*(
_output_shapes
:         АЪ
(sequential/dense/MatMul_7/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0░
sequential/dense/MatMul_7MatMul&sequential/dropout/Identity_7:output:00sequential/dense/MatMul_7/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЧ
)sequential/dense/BiasAdd_7/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0░
sequential/dense/BiasAdd_7BiasAdd#sequential/dense/MatMul_7:product:01sequential/dense/BiasAdd_7/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аw
sequential/dense/Relu_7Relu#sequential/dense/BiasAdd_7:output:0*
T0*(
_output_shapes
:         Аu
UnsortedSegmentMean_14/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧f
!UnsortedSegmentMean_14/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
UnsortedSegmentMean_14/onesFill%UnsortedSegmentMean_14/Shape:output:0*UnsortedSegmentMean_14/ones/Const:output:0*
T0*
_output_shapes
:╠
)UnsortedSegmentMean_14/UnsortedSegmentSumUnsortedSegmentSum$UnsortedSegmentMean_14/ones:output:0Squeeze_2:output:0
inputs_n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:t
*UnsortedSegmentMean_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
$UnsortedSegmentMean_14/strided_sliceStridedSlice
inputs_n_c3UnsortedSegmentMean_14/strided_slice/stack:output:05UnsortedSegmentMean_14/strided_slice/stack_1:output:05UnsortedSegmentMean_14/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask]
UnsortedSegmentMean_14/RankConst*
_output_shapes
: *
dtype0*
value	B :Z
UnsortedSegmentMean_14/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Р
UnsortedSegmentMean_14/subSub$UnsortedSegmentMean_14/Rank:output:0&UnsortedSegmentMean_14/Rank_1:output:0*
T0*
_output_shapes
: z
$UnsortedSegmentMean_14/ones_1/packedPackUnsortedSegmentMean_14/sub:z:0*
N*
T0*
_output_shapes
:e
#UnsortedSegmentMean_14/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R░
UnsortedSegmentMean_14/ones_1Fill-UnsortedSegmentMean_14/ones_1/packed:output:0,UnsortedSegmentMean_14/ones_1/Const:output:0*
T0	*#
_output_shapes
:         d
"UnsortedSegmentMean_14/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
UnsortedSegmentMean_14/concatConcatV2-UnsortedSegmentMean_14/strided_slice:output:0&UnsortedSegmentMean_14/ones_1:output:0+UnsortedSegmentMean_14/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         ╢
UnsortedSegmentMean_14/ReshapeReshape2UnsortedSegmentMean_14/UnsortedSegmentSum:output:0&UnsortedSegmentMean_14/concat:output:0*
Tshape0	*
T0*
_output_shapes
:e
 UnsortedSegmentMean_14/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
UnsortedSegmentMean_14/MaximumMaximum'UnsortedSegmentMean_14/Reshape:output:0)UnsortedSegmentMean_14/Maximum/y:output:0*
T0*
_output_shapes
:╧
+UnsortedSegmentMean_14/UnsortedSegmentSum_1UnsortedSegmentSum%sequential/dense/Relu_7:activations:0Squeeze_2:output:0
inputs_n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:ж
UnsortedSegmentMean_14/truedivRealDiv4UnsortedSegmentMean_14/UnsortedSegmentSum_1:output:0"UnsortedSegmentMean_14/Maximum:z:0*
T0*
_output_shapes
:R
GatherV2_30/axisConst*
_output_shapes
: *
dtype0*
value	B : к
GatherV2_30GatherV2update_connection/add_27:z:0Squeeze_3:output:0GatherV2_30/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_31/axisConst*
_output_shapes
: *
dtype0*
value	B : в
GatherV2_31GatherV2update_ip/add_27:z:0Squeeze_4:output:0GatherV2_31/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_35SqueezeGatherV2_31:output:0*
T0*
_output_shapes
:N

Squeeze_36SqueezeGatherV2_30:output:0*
T0*
_output_shapes
:P
concat_16/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_16ConcatV2Squeeze_36:output:0Squeeze_35:output:0concat_16/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_29EnsureShapeconcat_16:output:0*
T0*(
_output_shapes
:         А*
shape:         Аy
!sequential_1/dropout_1/Identity_7IdentityEnsureShape_29:output:0*
T0*(
_output_shapes
:         Ав
,sequential_1/dense_1/MatMul_7/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0╝
sequential_1/dense_1/MatMul_7MatMul*sequential_1/dropout_1/Identity_7:output:04sequential_1/dense_1/MatMul_7/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЯ
-sequential_1/dense_1/BiasAdd_7/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╝
sequential_1/dense_1/BiasAdd_7BiasAdd'sequential_1/dense_1/MatMul_7:product:05sequential_1/dense_1/BiasAdd_7/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А
sequential_1/dense_1/Relu_7Relu'sequential_1/dense_1/BiasAdd_7:output:0*
T0*(
_output_shapes
:         Аu
UnsortedSegmentMean_15/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧f
!UnsortedSegmentMean_15/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
UnsortedSegmentMean_15/onesFill%UnsortedSegmentMean_15/Shape:output:0*UnsortedSegmentMean_15/ones/Const:output:0*
T0*
_output_shapes
:╠
)UnsortedSegmentMean_15/UnsortedSegmentSumUnsortedSegmentSum$UnsortedSegmentMean_15/ones:output:0Squeeze_4:output:0
inputs_n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:t
*UnsortedSegmentMean_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:к
$UnsortedSegmentMean_15/strided_sliceStridedSlice
inputs_n_i3UnsortedSegmentMean_15/strided_slice/stack:output:05UnsortedSegmentMean_15/strided_slice/stack_1:output:05UnsortedSegmentMean_15/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask]
UnsortedSegmentMean_15/RankConst*
_output_shapes
: *
dtype0*
value	B :Z
UnsortedSegmentMean_15/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Р
UnsortedSegmentMean_15/subSub$UnsortedSegmentMean_15/Rank:output:0&UnsortedSegmentMean_15/Rank_1:output:0*
T0*
_output_shapes
: z
$UnsortedSegmentMean_15/ones_1/packedPackUnsortedSegmentMean_15/sub:z:0*
N*
T0*
_output_shapes
:e
#UnsortedSegmentMean_15/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R░
UnsortedSegmentMean_15/ones_1Fill-UnsortedSegmentMean_15/ones_1/packed:output:0,UnsortedSegmentMean_15/ones_1/Const:output:0*
T0	*#
_output_shapes
:         d
"UnsortedSegmentMean_15/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
UnsortedSegmentMean_15/concatConcatV2-UnsortedSegmentMean_15/strided_slice:output:0&UnsortedSegmentMean_15/ones_1:output:0+UnsortedSegmentMean_15/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         ╢
UnsortedSegmentMean_15/ReshapeReshape2UnsortedSegmentMean_15/UnsortedSegmentSum:output:0&UnsortedSegmentMean_15/concat:output:0*
Tshape0	*
T0*
_output_shapes
:e
 UnsortedSegmentMean_15/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
UnsortedSegmentMean_15/MaximumMaximum'UnsortedSegmentMean_15/Reshape:output:0)UnsortedSegmentMean_15/Maximum/y:output:0*
T0*
_output_shapes
:╙
+UnsortedSegmentMean_15/UnsortedSegmentSum_1UnsortedSegmentSum)sequential_1/dense_1/Relu_7:activations:0Squeeze_4:output:0
inputs_n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:ж
UnsortedSegmentMean_15/truedivRealDiv4UnsortedSegmentMean_15/UnsortedSegmentSum_1:output:0"UnsortedSegmentMean_15/Maximum:z:0*
T0*
_output_shapes
:У
EnsureShape_30EnsureShape"UnsortedSegmentMean_15/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         А}
update_ip/ReadVariableOp_7ReadVariableOp!update_ip_readvariableop_resource*
_output_shapes
:	А*
dtype0y
update_ip/unstack_7Unpack"update_ip/ReadVariableOp_7:value:0*
T0*"
_output_shapes
:А:А*	
numН
"update_ip/MatMul_14/ReadVariableOpReadVariableOp(update_ip_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Х
update_ip/MatMul_14MatMulEnsureShape_30:output:0*update_ip/MatMul_14/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АП
update_ip/BiasAdd_14BiasAddupdate_ip/MatMul_14:product:0update_ip/unstack_7:output:0*
T0*(
_output_shapes
:         Аg
update_ip/split_14/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ═
update_ip/split_14Split%update_ip/split_14/split_dim:output:0update_ip/BiasAdd_14:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitП
"update_ip/MatMul_15/ReadVariableOpReadVariableOp*update_ip_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Т
update_ip/MatMul_15MatMulupdate_ip/add_27:z:0*update_ip/MatMul_15/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АП
update_ip/BiasAdd_15BiasAddupdate_ip/MatMul_15:product:0update_ip/unstack_7:output:1*
T0*(
_output_shapes
:         Аf
update_ip/Const_7Const*
_output_shapes
:*
dtype0*!
valueB"А   А       g
update_ip/split_15/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Ў
update_ip/split_15SplitVupdate_ip/BiasAdd_15:output:0update_ip/Const_7:output:0%update_ip/split_15/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЖ
update_ip/add_28AddV2update_ip/split_14:output:0update_ip/split_15:output:0*
T0*(
_output_shapes
:         Аh
update_ip/Sigmoid_14Sigmoidupdate_ip/add_28:z:0*
T0*(
_output_shapes
:         АЖ
update_ip/add_29AddV2update_ip/split_14:output:1update_ip/split_15:output:1*
T0*(
_output_shapes
:         Аh
update_ip/Sigmoid_15Sigmoidupdate_ip/add_29:z:0*
T0*(
_output_shapes
:         АБ
update_ip/mul_21Mulupdate_ip/Sigmoid_15:y:0update_ip/split_15:output:2*
T0*(
_output_shapes
:         А
update_ip/add_30AddV2update_ip/split_14:output:2update_ip/mul_21:z:0*
T0*(
_output_shapes
:         Аa
update_ip/Tanh_7Tanhupdate_ip/add_30:z:0*
T0*(
_output_shapes
:         Аz
update_ip/mul_22Mulupdate_ip/Sigmoid_14:y:0update_ip/add_27:z:0*
T0*(
_output_shapes
:         АV
update_ip/sub_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?
update_ip/sub_7Subupdate_ip/sub_7/x:output:0update_ip/Sigmoid_14:y:0*
T0*(
_output_shapes
:         Аu
update_ip/mul_23Mulupdate_ip/sub_7:z:0update_ip/Tanh_7:y:0*
T0*(
_output_shapes
:         Аx
update_ip/add_31AddV2update_ip/mul_22:z:0update_ip/mul_23:z:0*
T0*(
_output_shapes
:         АУ
EnsureShape_31EnsureShape"UnsortedSegmentMean_14/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АН
"update_connection/ReadVariableOp_7ReadVariableOp)update_connection_readvariableop_resource*
_output_shapes
:	А*
dtype0Й
update_connection/unstack_7Unpack*update_connection/ReadVariableOp_7:value:0*
T0*"
_output_shapes
:А:А*	
numЭ
*update_connection/MatMul_14/ReadVariableOpReadVariableOp0update_connection_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0е
update_connection/MatMul_14MatMulEnsureShape_31:output:02update_connection/MatMul_14/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аз
update_connection/BiasAdd_14BiasAdd%update_connection/MatMul_14:product:0$update_connection/unstack_7:output:0*
T0*(
_output_shapes
:         Аo
$update_connection/split_14/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         х
update_connection/split_14Split-update_connection/split_14/split_dim:output:0%update_connection/BiasAdd_14:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЯ
*update_connection/MatMul_15/ReadVariableOpReadVariableOp2update_connection_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0к
update_connection/MatMul_15MatMulupdate_connection/add_27:z:02update_connection/MatMul_15/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аз
update_connection/BiasAdd_15BiasAdd%update_connection/MatMul_15:product:0$update_connection/unstack_7:output:1*
T0*(
_output_shapes
:         Аn
update_connection/Const_7Const*
_output_shapes
:*
dtype0*!
valueB"А   А       o
$update_connection/split_15/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         Ц
update_connection/split_15SplitV%update_connection/BiasAdd_15:output:0"update_connection/Const_7:output:0-update_connection/split_15/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitЮ
update_connection/add_28AddV2#update_connection/split_14:output:0#update_connection/split_15:output:0*
T0*(
_output_shapes
:         Аx
update_connection/Sigmoid_14Sigmoidupdate_connection/add_28:z:0*
T0*(
_output_shapes
:         АЮ
update_connection/add_29AddV2#update_connection/split_14:output:1#update_connection/split_15:output:1*
T0*(
_output_shapes
:         Аx
update_connection/Sigmoid_15Sigmoidupdate_connection/add_29:z:0*
T0*(
_output_shapes
:         АЩ
update_connection/mul_21Mul update_connection/Sigmoid_15:y:0#update_connection/split_15:output:2*
T0*(
_output_shapes
:         АЧ
update_connection/add_30AddV2#update_connection/split_14:output:2update_connection/mul_21:z:0*
T0*(
_output_shapes
:         Аq
update_connection/Tanh_7Tanhupdate_connection/add_30:z:0*
T0*(
_output_shapes
:         АТ
update_connection/mul_22Mul update_connection/Sigmoid_14:y:0update_connection/add_27:z:0*
T0*(
_output_shapes
:         А^
update_connection/sub_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ч
update_connection/sub_7Sub"update_connection/sub_7/x:output:0 update_connection/Sigmoid_14:y:0*
T0*(
_output_shapes
:         АН
update_connection/mul_23Mulupdate_connection/sub_7:z:0update_connection/Tanh_7:y:0*
T0*(
_output_shapes
:         АР
update_connection/add_31AddV2update_connection/mul_22:z:0update_connection/mul_23:z:0*
T0*(
_output_shapes
:         Аа
*sequential_2/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0к
sequential_2/dense_2/MatMulMatMulupdate_connection/add_31:z:02sequential_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЭ
+sequential_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0╢
sequential_2/dense_2/BiasAddBiasAdd%sequential_2/dense_2/MatMul:product:03sequential_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А{
sequential_2/dense_2/ReluRelu%sequential_2/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:         АЗ
sequential_2/dropout_2/IdentityIdentity'sequential_2/dense_2/Relu:activations:0*
T0*(
_output_shapes
:         АЯ
*sequential_2/dense_3/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_3_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0╡
sequential_2/dense_3/MatMulMatMul(sequential_2/dropout_2/Identity:output:02sequential_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ь
+sequential_2/dense_3/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0╡
sequential_2/dense_3/BiasAddBiasAdd%sequential_2/dense_3/MatMul:product:03sequential_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @z
sequential_2/dense_3/ReluRelu%sequential_2/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:         @Ж
sequential_2/dropout_3/IdentityIdentity'sequential_2/dense_3/Relu:activations:0*
T0*'
_output_shapes
:         @Ю
*sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0╡
sequential_2/dense_4/MatMulMatMul(sequential_2/dropout_3/Identity:output:02sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ь
+sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╡
sequential_2/dense_4/BiasAddBiasAdd%sequential_2/dense_4/MatMul:product:03sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         А
sequential_2/dense_4/SoftmaxSoftmax%sequential_2/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:         u
IdentityIdentity&sequential_2/dense_4/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         я
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/BiasAdd_1/ReadVariableOp*^sequential/dense/BiasAdd_2/ReadVariableOp*^sequential/dense/BiasAdd_3/ReadVariableOp*^sequential/dense/BiasAdd_4/ReadVariableOp*^sequential/dense/BiasAdd_5/ReadVariableOp*^sequential/dense/BiasAdd_6/ReadVariableOp*^sequential/dense/BiasAdd_7/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp)^sequential/dense/MatMul_1/ReadVariableOp)^sequential/dense/MatMul_2/ReadVariableOp)^sequential/dense/MatMul_3/ReadVariableOp)^sequential/dense/MatMul_4/ReadVariableOp)^sequential/dense/MatMul_5/ReadVariableOp)^sequential/dense/MatMul_6/ReadVariableOp)^sequential/dense/MatMul_7/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp.^sequential_1/dense_1/BiasAdd_1/ReadVariableOp.^sequential_1/dense_1/BiasAdd_2/ReadVariableOp.^sequential_1/dense_1/BiasAdd_3/ReadVariableOp.^sequential_1/dense_1/BiasAdd_4/ReadVariableOp.^sequential_1/dense_1/BiasAdd_5/ReadVariableOp.^sequential_1/dense_1/BiasAdd_6/ReadVariableOp.^sequential_1/dense_1/BiasAdd_7/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp-^sequential_1/dense_1/MatMul_1/ReadVariableOp-^sequential_1/dense_1/MatMul_2/ReadVariableOp-^sequential_1/dense_1/MatMul_3/ReadVariableOp-^sequential_1/dense_1/MatMul_4/ReadVariableOp-^sequential_1/dense_1/MatMul_5/ReadVariableOp-^sequential_1/dense_1/MatMul_6/ReadVariableOp-^sequential_1/dense_1/MatMul_7/ReadVariableOp,^sequential_2/dense_2/BiasAdd/ReadVariableOp+^sequential_2/dense_2/MatMul/ReadVariableOp,^sequential_2/dense_3/BiasAdd/ReadVariableOp+^sequential_2/dense_3/MatMul/ReadVariableOp,^sequential_2/dense_4/BiasAdd/ReadVariableOp+^sequential_2/dense_4/MatMul/ReadVariableOp(^update_connection/MatMul/ReadVariableOp*^update_connection/MatMul_1/ReadVariableOp+^update_connection/MatMul_10/ReadVariableOp+^update_connection/MatMul_11/ReadVariableOp+^update_connection/MatMul_12/ReadVariableOp+^update_connection/MatMul_13/ReadVariableOp+^update_connection/MatMul_14/ReadVariableOp+^update_connection/MatMul_15/ReadVariableOp*^update_connection/MatMul_2/ReadVariableOp*^update_connection/MatMul_3/ReadVariableOp*^update_connection/MatMul_4/ReadVariableOp*^update_connection/MatMul_5/ReadVariableOp*^update_connection/MatMul_6/ReadVariableOp*^update_connection/MatMul_7/ReadVariableOp*^update_connection/MatMul_8/ReadVariableOp*^update_connection/MatMul_9/ReadVariableOp!^update_connection/ReadVariableOp#^update_connection/ReadVariableOp_1#^update_connection/ReadVariableOp_2#^update_connection/ReadVariableOp_3#^update_connection/ReadVariableOp_4#^update_connection/ReadVariableOp_5#^update_connection/ReadVariableOp_6#^update_connection/ReadVariableOp_7 ^update_ip/MatMul/ReadVariableOp"^update_ip/MatMul_1/ReadVariableOp#^update_ip/MatMul_10/ReadVariableOp#^update_ip/MatMul_11/ReadVariableOp#^update_ip/MatMul_12/ReadVariableOp#^update_ip/MatMul_13/ReadVariableOp#^update_ip/MatMul_14/ReadVariableOp#^update_ip/MatMul_15/ReadVariableOp"^update_ip/MatMul_2/ReadVariableOp"^update_ip/MatMul_3/ReadVariableOp"^update_ip/MatMul_4/ReadVariableOp"^update_ip/MatMul_5/ReadVariableOp"^update_ip/MatMul_6/ReadVariableOp"^update_ip/MatMul_7/ReadVariableOp"^update_ip/MatMul_8/ReadVariableOp"^update_ip/MatMul_9/ReadVariableOp^update_ip/ReadVariableOp^update_ip/ReadVariableOp_1^update_ip/ReadVariableOp_2^update_ip/ReadVariableOp_3^update_ip/ReadVariableOp_4^update_ip/ReadVariableOp_5^update_ip/ReadVariableOp_6^update_ip/ReadVariableOp_7*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:::: : ::: : : : : : : : : : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/BiasAdd_1/ReadVariableOp)sequential/dense/BiasAdd_1/ReadVariableOp2V
)sequential/dense/BiasAdd_2/ReadVariableOp)sequential/dense/BiasAdd_2/ReadVariableOp2V
)sequential/dense/BiasAdd_3/ReadVariableOp)sequential/dense/BiasAdd_3/ReadVariableOp2V
)sequential/dense/BiasAdd_4/ReadVariableOp)sequential/dense/BiasAdd_4/ReadVariableOp2V
)sequential/dense/BiasAdd_5/ReadVariableOp)sequential/dense/BiasAdd_5/ReadVariableOp2V
)sequential/dense/BiasAdd_6/ReadVariableOp)sequential/dense/BiasAdd_6/ReadVariableOp2V
)sequential/dense/BiasAdd_7/ReadVariableOp)sequential/dense/BiasAdd_7/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2T
(sequential/dense/MatMul_1/ReadVariableOp(sequential/dense/MatMul_1/ReadVariableOp2T
(sequential/dense/MatMul_2/ReadVariableOp(sequential/dense/MatMul_2/ReadVariableOp2T
(sequential/dense/MatMul_3/ReadVariableOp(sequential/dense/MatMul_3/ReadVariableOp2T
(sequential/dense/MatMul_4/ReadVariableOp(sequential/dense/MatMul_4/ReadVariableOp2T
(sequential/dense/MatMul_5/ReadVariableOp(sequential/dense/MatMul_5/ReadVariableOp2T
(sequential/dense/MatMul_6/ReadVariableOp(sequential/dense/MatMul_6/ReadVariableOp2T
(sequential/dense/MatMul_7/ReadVariableOp(sequential/dense/MatMul_7/ReadVariableOp2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2^
-sequential_1/dense_1/BiasAdd_1/ReadVariableOp-sequential_1/dense_1/BiasAdd_1/ReadVariableOp2^
-sequential_1/dense_1/BiasAdd_2/ReadVariableOp-sequential_1/dense_1/BiasAdd_2/ReadVariableOp2^
-sequential_1/dense_1/BiasAdd_3/ReadVariableOp-sequential_1/dense_1/BiasAdd_3/ReadVariableOp2^
-sequential_1/dense_1/BiasAdd_4/ReadVariableOp-sequential_1/dense_1/BiasAdd_4/ReadVariableOp2^
-sequential_1/dense_1/BiasAdd_5/ReadVariableOp-sequential_1/dense_1/BiasAdd_5/ReadVariableOp2^
-sequential_1/dense_1/BiasAdd_6/ReadVariableOp-sequential_1/dense_1/BiasAdd_6/ReadVariableOp2^
-sequential_1/dense_1/BiasAdd_7/ReadVariableOp-sequential_1/dense_1/BiasAdd_7/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp2\
,sequential_1/dense_1/MatMul_1/ReadVariableOp,sequential_1/dense_1/MatMul_1/ReadVariableOp2\
,sequential_1/dense_1/MatMul_2/ReadVariableOp,sequential_1/dense_1/MatMul_2/ReadVariableOp2\
,sequential_1/dense_1/MatMul_3/ReadVariableOp,sequential_1/dense_1/MatMul_3/ReadVariableOp2\
,sequential_1/dense_1/MatMul_4/ReadVariableOp,sequential_1/dense_1/MatMul_4/ReadVariableOp2\
,sequential_1/dense_1/MatMul_5/ReadVariableOp,sequential_1/dense_1/MatMul_5/ReadVariableOp2\
,sequential_1/dense_1/MatMul_6/ReadVariableOp,sequential_1/dense_1/MatMul_6/ReadVariableOp2\
,sequential_1/dense_1/MatMul_7/ReadVariableOp,sequential_1/dense_1/MatMul_7/ReadVariableOp2Z
+sequential_2/dense_2/BiasAdd/ReadVariableOp+sequential_2/dense_2/BiasAdd/ReadVariableOp2X
*sequential_2/dense_2/MatMul/ReadVariableOp*sequential_2/dense_2/MatMul/ReadVariableOp2Z
+sequential_2/dense_3/BiasAdd/ReadVariableOp+sequential_2/dense_3/BiasAdd/ReadVariableOp2X
*sequential_2/dense_3/MatMul/ReadVariableOp*sequential_2/dense_3/MatMul/ReadVariableOp2Z
+sequential_2/dense_4/BiasAdd/ReadVariableOp+sequential_2/dense_4/BiasAdd/ReadVariableOp2X
*sequential_2/dense_4/MatMul/ReadVariableOp*sequential_2/dense_4/MatMul/ReadVariableOp2R
'update_connection/MatMul/ReadVariableOp'update_connection/MatMul/ReadVariableOp2V
)update_connection/MatMul_1/ReadVariableOp)update_connection/MatMul_1/ReadVariableOp2X
*update_connection/MatMul_10/ReadVariableOp*update_connection/MatMul_10/ReadVariableOp2X
*update_connection/MatMul_11/ReadVariableOp*update_connection/MatMul_11/ReadVariableOp2X
*update_connection/MatMul_12/ReadVariableOp*update_connection/MatMul_12/ReadVariableOp2X
*update_connection/MatMul_13/ReadVariableOp*update_connection/MatMul_13/ReadVariableOp2X
*update_connection/MatMul_14/ReadVariableOp*update_connection/MatMul_14/ReadVariableOp2X
*update_connection/MatMul_15/ReadVariableOp*update_connection/MatMul_15/ReadVariableOp2V
)update_connection/MatMul_2/ReadVariableOp)update_connection/MatMul_2/ReadVariableOp2V
)update_connection/MatMul_3/ReadVariableOp)update_connection/MatMul_3/ReadVariableOp2V
)update_connection/MatMul_4/ReadVariableOp)update_connection/MatMul_4/ReadVariableOp2V
)update_connection/MatMul_5/ReadVariableOp)update_connection/MatMul_5/ReadVariableOp2V
)update_connection/MatMul_6/ReadVariableOp)update_connection/MatMul_6/ReadVariableOp2V
)update_connection/MatMul_7/ReadVariableOp)update_connection/MatMul_7/ReadVariableOp2V
)update_connection/MatMul_8/ReadVariableOp)update_connection/MatMul_8/ReadVariableOp2V
)update_connection/MatMul_9/ReadVariableOp)update_connection/MatMul_9/ReadVariableOp2H
"update_connection/ReadVariableOp_1"update_connection/ReadVariableOp_12H
"update_connection/ReadVariableOp_2"update_connection/ReadVariableOp_22H
"update_connection/ReadVariableOp_3"update_connection/ReadVariableOp_32H
"update_connection/ReadVariableOp_4"update_connection/ReadVariableOp_42H
"update_connection/ReadVariableOp_5"update_connection/ReadVariableOp_52H
"update_connection/ReadVariableOp_6"update_connection/ReadVariableOp_62H
"update_connection/ReadVariableOp_7"update_connection/ReadVariableOp_72D
 update_connection/ReadVariableOp update_connection/ReadVariableOp2B
update_ip/MatMul/ReadVariableOpupdate_ip/MatMul/ReadVariableOp2F
!update_ip/MatMul_1/ReadVariableOp!update_ip/MatMul_1/ReadVariableOp2H
"update_ip/MatMul_10/ReadVariableOp"update_ip/MatMul_10/ReadVariableOp2H
"update_ip/MatMul_11/ReadVariableOp"update_ip/MatMul_11/ReadVariableOp2H
"update_ip/MatMul_12/ReadVariableOp"update_ip/MatMul_12/ReadVariableOp2H
"update_ip/MatMul_13/ReadVariableOp"update_ip/MatMul_13/ReadVariableOp2H
"update_ip/MatMul_14/ReadVariableOp"update_ip/MatMul_14/ReadVariableOp2H
"update_ip/MatMul_15/ReadVariableOp"update_ip/MatMul_15/ReadVariableOp2F
!update_ip/MatMul_2/ReadVariableOp!update_ip/MatMul_2/ReadVariableOp2F
!update_ip/MatMul_3/ReadVariableOp!update_ip/MatMul_3/ReadVariableOp2F
!update_ip/MatMul_4/ReadVariableOp!update_ip/MatMul_4/ReadVariableOp2F
!update_ip/MatMul_5/ReadVariableOp!update_ip/MatMul_5/ReadVariableOp2F
!update_ip/MatMul_6/ReadVariableOp!update_ip/MatMul_6/ReadVariableOp2F
!update_ip/MatMul_7/ReadVariableOp!update_ip/MatMul_7/ReadVariableOp2F
!update_ip/MatMul_8/ReadVariableOp!update_ip/MatMul_8/ReadVariableOp2F
!update_ip/MatMul_9/ReadVariableOp!update_ip/MatMul_9/ReadVariableOp28
update_ip/ReadVariableOp_1update_ip/ReadVariableOp_128
update_ip/ReadVariableOp_2update_ip/ReadVariableOp_228
update_ip/ReadVariableOp_3update_ip/ReadVariableOp_328
update_ip/ReadVariableOp_4update_ip/ReadVariableOp_428
update_ip/ReadVariableOp_5update_ip/ReadVariableOp_528
update_ip/ReadVariableOp_6update_ip/ReadVariableOp_628
update_ip/ReadVariableOp_7update_ip/ReadVariableOp_724
update_ip/ReadVariableOpupdate_ip/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:UQ

_output_shapes
:
5
_user_specified_nameinputs_src_ip_to_connection:UQ

_output_shapes
:
5
_user_specified_nameinputs_src_connection_to_ip:B>

_output_shapes
: 
$
_user_specified_name
inputs_n_i:B>

_output_shapes
: 
$
_user_specified_name
inputs_n_c:SO

_output_shapes
:
3
_user_specified_nameinputs_feature_connection:UQ

_output_shapes
:
5
_user_specified_nameinputs_dst_ip_to_connection:U Q

_output_shapes
:
5
_user_specified_nameinputs_dst_connection_to_ip
Л
т
N__inference_update_connection_layer_call_and_return_conditional_losses_4846701

inputs

states*
readvariableop_resource:	А2
matmul_readvariableop_resource:
АА4
 matmul_1_readvariableop_resource:
АА
identity

identity_1ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	А*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:А:А*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         АZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         АZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"А   А       \
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╔
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         АN
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         Аc
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         АR
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         А^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         АZ
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         АJ
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:         АT
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:         АJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         АR
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:         АW
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         АY
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         А[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         Аe
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         А:                  : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:XT
0
_output_shapes
:                  
 
_user_specified_namestates:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
ў
Щ
)__inference_dense_2_layer_call_fn_4847771

inputs
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_4845591p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	4847767:'#
!
_user_specified_name	4847765:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
б

e
F__inference_dropout_1_layer_call_and_return_conditional_losses_4847737

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╫

°
D__inference_dense_2_layer_call_and_return_conditional_losses_4845591

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         АS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╧

Ў
D__inference_dense_3_layer_call_and_return_conditional_losses_4845620

inputs1
matmul_readvariableop_resource:	А@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         @S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Ъ

e
F__inference_dropout_3_layer_call_and_return_conditional_losses_4847851

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         @Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ж
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    У
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:         @a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
▌
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_4845668

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Д
Я
.__inference_sequential_1_layer_call_fn_4845550
input_2
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_4845526p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	4845546:'#
!
_user_specified_name	4845544:Q M
(
_output_shapes
:         А
!
_user_specified_name	input_2
▌
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_4847742

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╨

ї
D__inference_dense_4_layer_call_and_return_conditional_losses_4847876

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
г
╚
G__inference_sequential_layer_call_and_return_conditional_losses_4845456
input_1!
dense_4845450:
АА
dense_4845452:	А
identityИвdense/StatefulPartitionedCall╣
dropout/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_4845448В
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_4845450dense_4845452*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_4845434v
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         АB
NoOpNoOp^dense/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:'#
!
_user_specified_name	4845452:'#
!
_user_specified_name	4845450:Q M
(
_output_shapes
:         А
!
_user_specified_name	input_1
щ
Ї
I__inference_sequential_1_layer_call_and_return_conditional_losses_4845526
input_2#
dense_1_4845520:
АА
dense_1_4845522:	А
identityИвdense_1/StatefulPartitionedCallв!dropout_1/StatefulPartitionedCall═
!dropout_1/StatefulPartitionedCallStatefulPartitionedCallinput_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_4845507Ф
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_4845520dense_1_4845522*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_4845519x
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         Аh
NoOpNoOp ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:'#
!
_user_specified_name	4845522:'#
!
_user_specified_name	4845520:Q M
(
_output_shapes
:         А
!
_user_specified_name	input_2
╧

Ў
D__inference_dense_3_layer_call_and_return_conditional_losses_4847829

inputs1
matmul_readvariableop_resource:	А@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:         @a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:         @S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
√
▄
F__inference_update_ip_layer_call_and_return_conditional_losses_4847562

inputs
states_0*
readvariableop_resource:	А2
matmul_readvariableop_resource:
АА4
 matmul_1_readvariableop_resource:
АА
identity

identity_1ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	А*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:А:А*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         АZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         АZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"А   А       \
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╔
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         АN
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         Аc
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         АR
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         А^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         АZ
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         АJ
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:         АV
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:         АJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         АR
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:         АW
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         АY
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         А[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         Аe
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         А:         А: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
(
_output_shapes
:         А
"
_user_specified_name
states_0:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
▓з
№q
 __inference__traced_save_4848660
file_prefix;
'read_disablecopyonread_update_ip_kernel:
ААG
3read_1_disablecopyonread_update_ip_recurrent_kernel:
АА:
'read_2_disablecopyonread_update_ip_bias:	АE
1read_3_disablecopyonread_update_connection_kernel:
ААO
;read_4_disablecopyonread_update_connection_recurrent_kernel:
ААB
/read_5_disablecopyonread_update_connection_bias:	А9
%read_6_disablecopyonread_dense_kernel:
АА2
#read_7_disablecopyonread_dense_bias:	А;
'read_8_disablecopyonread_dense_1_kernel:
АА4
%read_9_disablecopyonread_dense_1_bias:	А<
(read_10_disablecopyonread_dense_2_kernel:
АА5
&read_11_disablecopyonread_dense_2_bias:	А;
(read_12_disablecopyonread_dense_3_kernel:	А@4
&read_13_disablecopyonread_dense_3_bias:@:
(read_14_disablecopyonread_dense_4_kernel:@4
&read_15_disablecopyonread_dense_4_bias:-
#read_16_disablecopyonread_iteration:	 9
/read_17_disablecopyonread_current_learning_rate: E
1read_18_disablecopyonread_adam_m_update_ip_kernel:
ААE
1read_19_disablecopyonread_adam_v_update_ip_kernel:
ААO
;read_20_disablecopyonread_adam_m_update_ip_recurrent_kernel:
ААO
;read_21_disablecopyonread_adam_v_update_ip_recurrent_kernel:
ААB
/read_22_disablecopyonread_adam_m_update_ip_bias:	АB
/read_23_disablecopyonread_adam_v_update_ip_bias:	АM
9read_24_disablecopyonread_adam_m_update_connection_kernel:
ААM
9read_25_disablecopyonread_adam_v_update_connection_kernel:
ААW
Cread_26_disablecopyonread_adam_m_update_connection_recurrent_kernel:
ААW
Cread_27_disablecopyonread_adam_v_update_connection_recurrent_kernel:
ААJ
7read_28_disablecopyonread_adam_m_update_connection_bias:	АJ
7read_29_disablecopyonread_adam_v_update_connection_bias:	АA
-read_30_disablecopyonread_adam_m_dense_kernel:
ААA
-read_31_disablecopyonread_adam_v_dense_kernel:
АА:
+read_32_disablecopyonread_adam_m_dense_bias:	А:
+read_33_disablecopyonread_adam_v_dense_bias:	АC
/read_34_disablecopyonread_adam_m_dense_1_kernel:
ААC
/read_35_disablecopyonread_adam_v_dense_1_kernel:
АА<
-read_36_disablecopyonread_adam_m_dense_1_bias:	А<
-read_37_disablecopyonread_adam_v_dense_1_bias:	АC
/read_38_disablecopyonread_adam_m_dense_2_kernel:
ААC
/read_39_disablecopyonread_adam_v_dense_2_kernel:
АА<
-read_40_disablecopyonread_adam_m_dense_2_bias:	А<
-read_41_disablecopyonread_adam_v_dense_2_bias:	АB
/read_42_disablecopyonread_adam_m_dense_3_kernel:	А@B
/read_43_disablecopyonread_adam_v_dense_3_kernel:	А@;
-read_44_disablecopyonread_adam_m_dense_3_bias:@;
-read_45_disablecopyonread_adam_v_dense_3_bias:@A
/read_46_disablecopyonread_adam_m_dense_4_kernel:@A
/read_47_disablecopyonread_adam_v_dense_4_kernel:@;
-read_48_disablecopyonread_adam_m_dense_4_bias:;
-read_49_disablecopyonread_adam_v_dense_4_bias:+
!read_50_disablecopyonread_total_1: +
!read_51_disablecopyonread_count_1: )
read_52_disablecopyonread_total: )
read_53_disablecopyonread_count: :
+read_54_disablecopyonread_true_positives_32:	╚7
(read_55_disablecopyonread_true_negatives:	╚;
,read_56_disablecopyonread_false_positives_17:	╚;
,read_57_disablecopyonread_false_negatives_17:	╚9
+read_58_disablecopyonread_true_positives_31::
,read_59_disablecopyonread_false_negatives_16:9
+read_60_disablecopyonread_true_positives_30::
,read_61_disablecopyonread_false_positives_16:9
+read_62_disablecopyonread_true_positives_29::
,read_63_disablecopyonread_false_negatives_15:9
+read_64_disablecopyonread_true_positives_28::
,read_65_disablecopyonread_false_positives_15:9
+read_66_disablecopyonread_true_positives_27::
,read_67_disablecopyonread_false_negatives_14:9
+read_68_disablecopyonread_true_positives_26::
,read_69_disablecopyonread_false_positives_14:9
+read_70_disablecopyonread_true_positives_25::
,read_71_disablecopyonread_false_negatives_13:9
+read_72_disablecopyonread_true_positives_24::
,read_73_disablecopyonread_false_positives_13:9
+read_74_disablecopyonread_true_positives_23::
,read_75_disablecopyonread_false_negatives_12:9
+read_76_disablecopyonread_true_positives_22::
,read_77_disablecopyonread_false_positives_12:9
+read_78_disablecopyonread_true_positives_21::
,read_79_disablecopyonread_false_negatives_11:9
+read_80_disablecopyonread_true_positives_20::
,read_81_disablecopyonread_false_positives_11:9
+read_82_disablecopyonread_true_positives_19::
,read_83_disablecopyonread_false_negatives_10:9
+read_84_disablecopyonread_true_positives_18::
,read_85_disablecopyonread_false_positives_10:9
+read_86_disablecopyonread_true_positives_17:9
+read_87_disablecopyonread_false_negatives_9:9
+read_88_disablecopyonread_true_positives_16:9
+read_89_disablecopyonread_false_positives_9:9
+read_90_disablecopyonread_true_positives_15:9
+read_91_disablecopyonread_false_negatives_8:9
+read_92_disablecopyonread_true_positives_14:9
+read_93_disablecopyonread_false_positives_8:9
+read_94_disablecopyonread_true_positives_13:9
+read_95_disablecopyonread_false_negatives_7:9
+read_96_disablecopyonread_true_positives_12:9
+read_97_disablecopyonread_false_positives_7:9
+read_98_disablecopyonread_true_positives_11:9
+read_99_disablecopyonread_false_negatives_6::
,read_100_disablecopyonread_true_positives_10::
,read_101_disablecopyonread_false_positives_6:9
+read_102_disablecopyonread_true_positives_9::
,read_103_disablecopyonread_false_negatives_5:9
+read_104_disablecopyonread_true_positives_8::
,read_105_disablecopyonread_false_positives_5:9
+read_106_disablecopyonread_true_positives_7::
,read_107_disablecopyonread_false_negatives_4:9
+read_108_disablecopyonread_true_positives_6::
,read_109_disablecopyonread_false_positives_4:9
+read_110_disablecopyonread_true_positives_5::
,read_111_disablecopyonread_false_negatives_3:9
+read_112_disablecopyonread_true_positives_4::
,read_113_disablecopyonread_false_positives_3:9
+read_114_disablecopyonread_true_positives_3::
,read_115_disablecopyonread_false_negatives_2:9
+read_116_disablecopyonread_true_positives_2::
,read_117_disablecopyonread_false_positives_2:9
+read_118_disablecopyonread_true_positives_1::
,read_119_disablecopyonread_false_positives_1::
,read_120_disablecopyonread_false_negatives_1:?
1read_121_disablecopyonread_weights_intermediate_1:7
)read_122_disablecopyonread_true_positives:8
*read_123_disablecopyonread_false_positives:8
*read_124_disablecopyonread_false_negatives:=
/read_125_disablecopyonread_weights_intermediate:
savev2_const
identity_253ИвMergeV2CheckpointsвRead/DisableCopyOnReadвRead/ReadVariableOpвRead_1/DisableCopyOnReadвRead_1/ReadVariableOpвRead_10/DisableCopyOnReadвRead_10/ReadVariableOpвRead_100/DisableCopyOnReadвRead_100/ReadVariableOpвRead_101/DisableCopyOnReadвRead_101/ReadVariableOpвRead_102/DisableCopyOnReadвRead_102/ReadVariableOpвRead_103/DisableCopyOnReadвRead_103/ReadVariableOpвRead_104/DisableCopyOnReadвRead_104/ReadVariableOpвRead_105/DisableCopyOnReadвRead_105/ReadVariableOpвRead_106/DisableCopyOnReadвRead_106/ReadVariableOpвRead_107/DisableCopyOnReadвRead_107/ReadVariableOpвRead_108/DisableCopyOnReadвRead_108/ReadVariableOpвRead_109/DisableCopyOnReadвRead_109/ReadVariableOpвRead_11/DisableCopyOnReadвRead_11/ReadVariableOpвRead_110/DisableCopyOnReadвRead_110/ReadVariableOpвRead_111/DisableCopyOnReadвRead_111/ReadVariableOpвRead_112/DisableCopyOnReadвRead_112/ReadVariableOpвRead_113/DisableCopyOnReadвRead_113/ReadVariableOpвRead_114/DisableCopyOnReadвRead_114/ReadVariableOpвRead_115/DisableCopyOnReadвRead_115/ReadVariableOpвRead_116/DisableCopyOnReadвRead_116/ReadVariableOpвRead_117/DisableCopyOnReadвRead_117/ReadVariableOpвRead_118/DisableCopyOnReadвRead_118/ReadVariableOpвRead_119/DisableCopyOnReadвRead_119/ReadVariableOpвRead_12/DisableCopyOnReadвRead_12/ReadVariableOpвRead_120/DisableCopyOnReadвRead_120/ReadVariableOpвRead_121/DisableCopyOnReadвRead_121/ReadVariableOpвRead_122/DisableCopyOnReadвRead_122/ReadVariableOpвRead_123/DisableCopyOnReadвRead_123/ReadVariableOpвRead_124/DisableCopyOnReadвRead_124/ReadVariableOpвRead_125/DisableCopyOnReadвRead_125/ReadVariableOpвRead_13/DisableCopyOnReadвRead_13/ReadVariableOpвRead_14/DisableCopyOnReadвRead_14/ReadVariableOpвRead_15/DisableCopyOnReadвRead_15/ReadVariableOpвRead_16/DisableCopyOnReadвRead_16/ReadVariableOpвRead_17/DisableCopyOnReadвRead_17/ReadVariableOpвRead_18/DisableCopyOnReadвRead_18/ReadVariableOpвRead_19/DisableCopyOnReadвRead_19/ReadVariableOpвRead_2/DisableCopyOnReadвRead_2/ReadVariableOpвRead_20/DisableCopyOnReadвRead_20/ReadVariableOpвRead_21/DisableCopyOnReadвRead_21/ReadVariableOpвRead_22/DisableCopyOnReadвRead_22/ReadVariableOpвRead_23/DisableCopyOnReadвRead_23/ReadVariableOpвRead_24/DisableCopyOnReadвRead_24/ReadVariableOpвRead_25/DisableCopyOnReadвRead_25/ReadVariableOpвRead_26/DisableCopyOnReadвRead_26/ReadVariableOpвRead_27/DisableCopyOnReadвRead_27/ReadVariableOpвRead_28/DisableCopyOnReadвRead_28/ReadVariableOpвRead_29/DisableCopyOnReadвRead_29/ReadVariableOpвRead_3/DisableCopyOnReadвRead_3/ReadVariableOpвRead_30/DisableCopyOnReadвRead_30/ReadVariableOpвRead_31/DisableCopyOnReadвRead_31/ReadVariableOpвRead_32/DisableCopyOnReadвRead_32/ReadVariableOpвRead_33/DisableCopyOnReadвRead_33/ReadVariableOpвRead_34/DisableCopyOnReadвRead_34/ReadVariableOpвRead_35/DisableCopyOnReadвRead_35/ReadVariableOpвRead_36/DisableCopyOnReadвRead_36/ReadVariableOpвRead_37/DisableCopyOnReadвRead_37/ReadVariableOpвRead_38/DisableCopyOnReadвRead_38/ReadVariableOpвRead_39/DisableCopyOnReadвRead_39/ReadVariableOpвRead_4/DisableCopyOnReadвRead_4/ReadVariableOpвRead_40/DisableCopyOnReadвRead_40/ReadVariableOpвRead_41/DisableCopyOnReadвRead_41/ReadVariableOpвRead_42/DisableCopyOnReadвRead_42/ReadVariableOpвRead_43/DisableCopyOnReadвRead_43/ReadVariableOpвRead_44/DisableCopyOnReadвRead_44/ReadVariableOpвRead_45/DisableCopyOnReadвRead_45/ReadVariableOpвRead_46/DisableCopyOnReadвRead_46/ReadVariableOpвRead_47/DisableCopyOnReadвRead_47/ReadVariableOpвRead_48/DisableCopyOnReadвRead_48/ReadVariableOpвRead_49/DisableCopyOnReadвRead_49/ReadVariableOpвRead_5/DisableCopyOnReadвRead_5/ReadVariableOpвRead_50/DisableCopyOnReadвRead_50/ReadVariableOpвRead_51/DisableCopyOnReadвRead_51/ReadVariableOpвRead_52/DisableCopyOnReadвRead_52/ReadVariableOpвRead_53/DisableCopyOnReadвRead_53/ReadVariableOpвRead_54/DisableCopyOnReadвRead_54/ReadVariableOpвRead_55/DisableCopyOnReadвRead_55/ReadVariableOpвRead_56/DisableCopyOnReadвRead_56/ReadVariableOpвRead_57/DisableCopyOnReadвRead_57/ReadVariableOpвRead_58/DisableCopyOnReadвRead_58/ReadVariableOpвRead_59/DisableCopyOnReadвRead_59/ReadVariableOpвRead_6/DisableCopyOnReadвRead_6/ReadVariableOpвRead_60/DisableCopyOnReadвRead_60/ReadVariableOpвRead_61/DisableCopyOnReadвRead_61/ReadVariableOpвRead_62/DisableCopyOnReadвRead_62/ReadVariableOpвRead_63/DisableCopyOnReadвRead_63/ReadVariableOpвRead_64/DisableCopyOnReadвRead_64/ReadVariableOpвRead_65/DisableCopyOnReadвRead_65/ReadVariableOpвRead_66/DisableCopyOnReadвRead_66/ReadVariableOpвRead_67/DisableCopyOnReadвRead_67/ReadVariableOpвRead_68/DisableCopyOnReadвRead_68/ReadVariableOpвRead_69/DisableCopyOnReadвRead_69/ReadVariableOpвRead_7/DisableCopyOnReadвRead_7/ReadVariableOpвRead_70/DisableCopyOnReadвRead_70/ReadVariableOpвRead_71/DisableCopyOnReadвRead_71/ReadVariableOpвRead_72/DisableCopyOnReadвRead_72/ReadVariableOpвRead_73/DisableCopyOnReadвRead_73/ReadVariableOpвRead_74/DisableCopyOnReadвRead_74/ReadVariableOpвRead_75/DisableCopyOnReadвRead_75/ReadVariableOpвRead_76/DisableCopyOnReadвRead_76/ReadVariableOpвRead_77/DisableCopyOnReadвRead_77/ReadVariableOpвRead_78/DisableCopyOnReadвRead_78/ReadVariableOpвRead_79/DisableCopyOnReadвRead_79/ReadVariableOpвRead_8/DisableCopyOnReadвRead_8/ReadVariableOpвRead_80/DisableCopyOnReadвRead_80/ReadVariableOpвRead_81/DisableCopyOnReadвRead_81/ReadVariableOpвRead_82/DisableCopyOnReadвRead_82/ReadVariableOpвRead_83/DisableCopyOnReadвRead_83/ReadVariableOpвRead_84/DisableCopyOnReadвRead_84/ReadVariableOpвRead_85/DisableCopyOnReadвRead_85/ReadVariableOpвRead_86/DisableCopyOnReadвRead_86/ReadVariableOpвRead_87/DisableCopyOnReadвRead_87/ReadVariableOpвRead_88/DisableCopyOnReadвRead_88/ReadVariableOpвRead_89/DisableCopyOnReadвRead_89/ReadVariableOpвRead_9/DisableCopyOnReadвRead_9/ReadVariableOpвRead_90/DisableCopyOnReadвRead_90/ReadVariableOpвRead_91/DisableCopyOnReadвRead_91/ReadVariableOpвRead_92/DisableCopyOnReadвRead_92/ReadVariableOpвRead_93/DisableCopyOnReadвRead_93/ReadVariableOpвRead_94/DisableCopyOnReadвRead_94/ReadVariableOpвRead_95/DisableCopyOnReadвRead_95/ReadVariableOpвRead_96/DisableCopyOnReadвRead_96/ReadVariableOpвRead_97/DisableCopyOnReadвRead_97/ReadVariableOpвRead_98/DisableCopyOnReadвRead_98/ReadVariableOpвRead_99/DisableCopyOnReadвRead_99/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: y
Read/DisableCopyOnReadDisableCopyOnRead'read_disablecopyonread_update_ip_kernel"/device:CPU:0*
_output_shapes
 е
Read/ReadVariableOpReadVariableOp'read_disablecopyonread_update_ip_kernel^Read/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0k
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААc

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААЗ
Read_1/DisableCopyOnReadDisableCopyOnRead3read_1_disablecopyonread_update_ip_recurrent_kernel"/device:CPU:0*
_output_shapes
 ╡
Read_1/ReadVariableOpReadVariableOp3read_1_disablecopyonread_update_ip_recurrent_kernel^Read_1/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0o

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААe

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0* 
_output_shapes
:
АА{
Read_2/DisableCopyOnReadDisableCopyOnRead'read_2_disablecopyonread_update_ip_bias"/device:CPU:0*
_output_shapes
 и
Read_2/ReadVariableOpReadVariableOp'read_2_disablecopyonread_update_ip_bias^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А*
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Аd

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	АЕ
Read_3/DisableCopyOnReadDisableCopyOnRead1read_3_disablecopyonread_update_connection_kernel"/device:CPU:0*
_output_shapes
 │
Read_3/ReadVariableOpReadVariableOp1read_3_disablecopyonread_update_connection_kernel^Read_3/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0o

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААe

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААП
Read_4/DisableCopyOnReadDisableCopyOnRead;read_4_disablecopyonread_update_connection_recurrent_kernel"/device:CPU:0*
_output_shapes
 ╜
Read_4/ReadVariableOpReadVariableOp;read_4_disablecopyonread_update_connection_recurrent_kernel^Read_4/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0o

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААe

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААГ
Read_5/DisableCopyOnReadDisableCopyOnRead/read_5_disablecopyonread_update_connection_bias"/device:CPU:0*
_output_shapes
 ░
Read_5/ReadVariableOpReadVariableOp/read_5_disablecopyonread_update_connection_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А*
dtype0o
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Аf
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:	Аy
Read_6/DisableCopyOnReadDisableCopyOnRead%read_6_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 з
Read_6/ReadVariableOpReadVariableOp%read_6_disablecopyonread_dense_kernel^Read_6/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0p
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААw
Read_7/DisableCopyOnReadDisableCopyOnRead#read_7_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 а
Read_7/ReadVariableOpReadVariableOp#read_7_disablecopyonread_dense_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:А{
Read_8/DisableCopyOnReadDisableCopyOnRead'read_8_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 й
Read_8/ReadVariableOpReadVariableOp'read_8_disablecopyonread_dense_1_kernel^Read_8/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0p
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААy
Read_9/DisableCopyOnReadDisableCopyOnRead%read_9_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 в
Read_9/ReadVariableOpReadVariableOp%read_9_disablecopyonread_dense_1_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0k
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:А}
Read_10/DisableCopyOnReadDisableCopyOnRead(read_10_disablecopyonread_dense_2_kernel"/device:CPU:0*
_output_shapes
 м
Read_10/ReadVariableOpReadVariableOp(read_10_disablecopyonread_dense_2_kernel^Read_10/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0* 
_output_shapes
:
АА{
Read_11/DisableCopyOnReadDisableCopyOnRead&read_11_disablecopyonread_dense_2_bias"/device:CPU:0*
_output_shapes
 е
Read_11/ReadVariableOpReadVariableOp&read_11_disablecopyonread_dense_2_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:А}
Read_12/DisableCopyOnReadDisableCopyOnRead(read_12_disablecopyonread_dense_3_kernel"/device:CPU:0*
_output_shapes
 л
Read_12/ReadVariableOpReadVariableOp(read_12_disablecopyonread_dense_3_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А@*
dtype0p
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	А@f
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:	А@{
Read_13/DisableCopyOnReadDisableCopyOnRead&read_13_disablecopyonread_dense_3_bias"/device:CPU:0*
_output_shapes
 д
Read_13/ReadVariableOpReadVariableOp&read_13_disablecopyonread_dense_3_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:@}
Read_14/DisableCopyOnReadDisableCopyOnRead(read_14_disablecopyonread_dense_4_kernel"/device:CPU:0*
_output_shapes
 к
Read_14/ReadVariableOpReadVariableOp(read_14_disablecopyonread_dense_4_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes

:@{
Read_15/DisableCopyOnReadDisableCopyOnRead&read_15_disablecopyonread_dense_4_bias"/device:CPU:0*
_output_shapes
 д
Read_15/ReadVariableOpReadVariableOp&read_15_disablecopyonread_dense_4_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_16/DisableCopyOnReadDisableCopyOnRead#read_16_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 Э
Read_16/ReadVariableOpReadVariableOp#read_16_disablecopyonread_iteration^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0	*
_output_shapes
: Д
Read_17/DisableCopyOnReadDisableCopyOnRead/read_17_disablecopyonread_current_learning_rate"/device:CPU:0*
_output_shapes
 й
Read_17/ReadVariableOpReadVariableOp/read_17_disablecopyonread_current_learning_rate^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
: Ж
Read_18/DisableCopyOnReadDisableCopyOnRead1read_18_disablecopyonread_adam_m_update_ip_kernel"/device:CPU:0*
_output_shapes
 ╡
Read_18/ReadVariableOpReadVariableOp1read_18_disablecopyonread_adam_m_update_ip_kernel^Read_18/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААЖ
Read_19/DisableCopyOnReadDisableCopyOnRead1read_19_disablecopyonread_adam_v_update_ip_kernel"/device:CPU:0*
_output_shapes
 ╡
Read_19/ReadVariableOpReadVariableOp1read_19_disablecopyonread_adam_v_update_ip_kernel^Read_19/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААР
Read_20/DisableCopyOnReadDisableCopyOnRead;read_20_disablecopyonread_adam_m_update_ip_recurrent_kernel"/device:CPU:0*
_output_shapes
 ┐
Read_20/ReadVariableOpReadVariableOp;read_20_disablecopyonread_adam_m_update_ip_recurrent_kernel^Read_20/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААР
Read_21/DisableCopyOnReadDisableCopyOnRead;read_21_disablecopyonread_adam_v_update_ip_recurrent_kernel"/device:CPU:0*
_output_shapes
 ┐
Read_21/ReadVariableOpReadVariableOp;read_21_disablecopyonread_adam_v_update_ip_recurrent_kernel^Read_21/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААД
Read_22/DisableCopyOnReadDisableCopyOnRead/read_22_disablecopyonread_adam_m_update_ip_bias"/device:CPU:0*
_output_shapes
 ▓
Read_22/ReadVariableOpReadVariableOp/read_22_disablecopyonread_adam_m_update_ip_bias^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А*
dtype0p
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Аf
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:	АД
Read_23/DisableCopyOnReadDisableCopyOnRead/read_23_disablecopyonread_adam_v_update_ip_bias"/device:CPU:0*
_output_shapes
 ▓
Read_23/ReadVariableOpReadVariableOp/read_23_disablecopyonread_adam_v_update_ip_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А*
dtype0p
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Аf
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:	АО
Read_24/DisableCopyOnReadDisableCopyOnRead9read_24_disablecopyonread_adam_m_update_connection_kernel"/device:CPU:0*
_output_shapes
 ╜
Read_24/ReadVariableOpReadVariableOp9read_24_disablecopyonread_adam_m_update_connection_kernel^Read_24/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААО
Read_25/DisableCopyOnReadDisableCopyOnRead9read_25_disablecopyonread_adam_v_update_connection_kernel"/device:CPU:0*
_output_shapes
 ╜
Read_25/ReadVariableOpReadVariableOp9read_25_disablecopyonread_adam_v_update_connection_kernel^Read_25/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААШ
Read_26/DisableCopyOnReadDisableCopyOnReadCread_26_disablecopyonread_adam_m_update_connection_recurrent_kernel"/device:CPU:0*
_output_shapes
 ╟
Read_26/ReadVariableOpReadVariableOpCread_26_disablecopyonread_adam_m_update_connection_recurrent_kernel^Read_26/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААШ
Read_27/DisableCopyOnReadDisableCopyOnReadCread_27_disablecopyonread_adam_v_update_connection_recurrent_kernel"/device:CPU:0*
_output_shapes
 ╟
Read_27/ReadVariableOpReadVariableOpCread_27_disablecopyonread_adam_v_update_connection_recurrent_kernel^Read_27/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААМ
Read_28/DisableCopyOnReadDisableCopyOnRead7read_28_disablecopyonread_adam_m_update_connection_bias"/device:CPU:0*
_output_shapes
 ║
Read_28/ReadVariableOpReadVariableOp7read_28_disablecopyonread_adam_m_update_connection_bias^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А*
dtype0p
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Аf
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:	АМ
Read_29/DisableCopyOnReadDisableCopyOnRead7read_29_disablecopyonread_adam_v_update_connection_bias"/device:CPU:0*
_output_shapes
 ║
Read_29/ReadVariableOpReadVariableOp7read_29_disablecopyonread_adam_v_update_connection_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А*
dtype0p
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Аf
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:	АВ
Read_30/DisableCopyOnReadDisableCopyOnRead-read_30_disablecopyonread_adam_m_dense_kernel"/device:CPU:0*
_output_shapes
 ▒
Read_30/ReadVariableOpReadVariableOp-read_30_disablecopyonread_adam_m_dense_kernel^Read_30/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААВ
Read_31/DisableCopyOnReadDisableCopyOnRead-read_31_disablecopyonread_adam_v_dense_kernel"/device:CPU:0*
_output_shapes
 ▒
Read_31/ReadVariableOpReadVariableOp-read_31_disablecopyonread_adam_v_dense_kernel^Read_31/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААА
Read_32/DisableCopyOnReadDisableCopyOnRead+read_32_disablecopyonread_adam_m_dense_bias"/device:CPU:0*
_output_shapes
 к
Read_32/ReadVariableOpReadVariableOp+read_32_disablecopyonread_adam_m_dense_bias^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes	
:АА
Read_33/DisableCopyOnReadDisableCopyOnRead+read_33_disablecopyonread_adam_v_dense_bias"/device:CPU:0*
_output_shapes
 к
Read_33/ReadVariableOpReadVariableOp+read_33_disablecopyonread_adam_v_dense_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes	
:АД
Read_34/DisableCopyOnReadDisableCopyOnRead/read_34_disablecopyonread_adam_m_dense_1_kernel"/device:CPU:0*
_output_shapes
 │
Read_34/ReadVariableOpReadVariableOp/read_34_disablecopyonread_adam_m_dense_1_kernel^Read_34/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААД
Read_35/DisableCopyOnReadDisableCopyOnRead/read_35_disablecopyonread_adam_v_dense_1_kernel"/device:CPU:0*
_output_shapes
 │
Read_35/ReadVariableOpReadVariableOp/read_35_disablecopyonread_adam_v_dense_1_kernel^Read_35/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААВ
Read_36/DisableCopyOnReadDisableCopyOnRead-read_36_disablecopyonread_adam_m_dense_1_bias"/device:CPU:0*
_output_shapes
 м
Read_36/ReadVariableOpReadVariableOp-read_36_disablecopyonread_adam_m_dense_1_bias^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes	
:АВ
Read_37/DisableCopyOnReadDisableCopyOnRead-read_37_disablecopyonread_adam_v_dense_1_bias"/device:CPU:0*
_output_shapes
 м
Read_37/ReadVariableOpReadVariableOp-read_37_disablecopyonread_adam_v_dense_1_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes	
:АД
Read_38/DisableCopyOnReadDisableCopyOnRead/read_38_disablecopyonread_adam_m_dense_2_kernel"/device:CPU:0*
_output_shapes
 │
Read_38/ReadVariableOpReadVariableOp/read_38_disablecopyonread_adam_m_dense_2_kernel^Read_38/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААД
Read_39/DisableCopyOnReadDisableCopyOnRead/read_39_disablecopyonread_adam_v_dense_2_kernel"/device:CPU:0*
_output_shapes
 │
Read_39/ReadVariableOpReadVariableOp/read_39_disablecopyonread_adam_v_dense_2_kernel^Read_39/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
АА*
dtype0q
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ААg
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ААВ
Read_40/DisableCopyOnReadDisableCopyOnRead-read_40_disablecopyonread_adam_m_dense_2_bias"/device:CPU:0*
_output_shapes
 м
Read_40/ReadVariableOpReadVariableOp-read_40_disablecopyonread_adam_m_dense_2_bias^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes	
:АВ
Read_41/DisableCopyOnReadDisableCopyOnRead-read_41_disablecopyonread_adam_v_dense_2_bias"/device:CPU:0*
_output_shapes
 м
Read_41/ReadVariableOpReadVariableOp-read_41_disablecopyonread_adam_v_dense_2_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes	
:АД
Read_42/DisableCopyOnReadDisableCopyOnRead/read_42_disablecopyonread_adam_m_dense_3_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_42/ReadVariableOpReadVariableOp/read_42_disablecopyonread_adam_m_dense_3_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А@*
dtype0p
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	А@f
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
:	А@Д
Read_43/DisableCopyOnReadDisableCopyOnRead/read_43_disablecopyonread_adam_v_dense_3_kernel"/device:CPU:0*
_output_shapes
 ▓
Read_43/ReadVariableOpReadVariableOp/read_43_disablecopyonread_adam_v_dense_3_kernel^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А@*
dtype0p
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	А@f
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:	А@В
Read_44/DisableCopyOnReadDisableCopyOnRead-read_44_disablecopyonread_adam_m_dense_3_bias"/device:CPU:0*
_output_shapes
 л
Read_44/ReadVariableOpReadVariableOp-read_44_disablecopyonread_adam_m_dense_3_bias^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
:@В
Read_45/DisableCopyOnReadDisableCopyOnRead-read_45_disablecopyonread_adam_v_dense_3_bias"/device:CPU:0*
_output_shapes
 л
Read_45/ReadVariableOpReadVariableOp-read_45_disablecopyonread_adam_v_dense_3_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:@Д
Read_46/DisableCopyOnReadDisableCopyOnRead/read_46_disablecopyonread_adam_m_dense_4_kernel"/device:CPU:0*
_output_shapes
 ▒
Read_46/ReadVariableOpReadVariableOp/read_46_disablecopyonread_adam_m_dense_4_kernel^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes

:@Д
Read_47/DisableCopyOnReadDisableCopyOnRead/read_47_disablecopyonread_adam_v_dense_4_kernel"/device:CPU:0*
_output_shapes
 ▒
Read_47/ReadVariableOpReadVariableOp/read_47_disablecopyonread_adam_v_dense_4_kernel^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes

:@В
Read_48/DisableCopyOnReadDisableCopyOnRead-read_48_disablecopyonread_adam_m_dense_4_bias"/device:CPU:0*
_output_shapes
 л
Read_48/ReadVariableOpReadVariableOp-read_48_disablecopyonread_adam_m_dense_4_bias^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes
:В
Read_49/DisableCopyOnReadDisableCopyOnRead-read_49_disablecopyonread_adam_v_dense_4_bias"/device:CPU:0*
_output_shapes
 л
Read_49/ReadVariableOpReadVariableOp-read_49_disablecopyonread_adam_v_dense_4_bias^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_50/DisableCopyOnReadDisableCopyOnRead!read_50_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 Ы
Read_50/ReadVariableOpReadVariableOp!read_50_disablecopyonread_total_1^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_51/DisableCopyOnReadDisableCopyOnRead!read_51_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 Ы
Read_51/ReadVariableOpReadVariableOp!read_51_disablecopyonread_count_1^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_52/DisableCopyOnReadDisableCopyOnReadread_52_disablecopyonread_total"/device:CPU:0*
_output_shapes
 Щ
Read_52/ReadVariableOpReadVariableOpread_52_disablecopyonread_total^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_53/DisableCopyOnReadDisableCopyOnReadread_53_disablecopyonread_count"/device:CPU:0*
_output_shapes
 Щ
Read_53/ReadVariableOpReadVariableOpread_53_disablecopyonread_count^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
: А
Read_54/DisableCopyOnReadDisableCopyOnRead+read_54_disablecopyonread_true_positives_32"/device:CPU:0*
_output_shapes
 к
Read_54/ReadVariableOpReadVariableOp+read_54_disablecopyonread_true_positives_32^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:╚*
dtype0m
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:╚d
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes	
:╚}
Read_55/DisableCopyOnReadDisableCopyOnRead(read_55_disablecopyonread_true_negatives"/device:CPU:0*
_output_shapes
 з
Read_55/ReadVariableOpReadVariableOp(read_55_disablecopyonread_true_negatives^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:╚*
dtype0m
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:╚d
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes	
:╚Б
Read_56/DisableCopyOnReadDisableCopyOnRead,read_56_disablecopyonread_false_positives_17"/device:CPU:0*
_output_shapes
 л
Read_56/ReadVariableOpReadVariableOp,read_56_disablecopyonread_false_positives_17^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:╚*
dtype0m
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:╚d
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes	
:╚Б
Read_57/DisableCopyOnReadDisableCopyOnRead,read_57_disablecopyonread_false_negatives_17"/device:CPU:0*
_output_shapes
 л
Read_57/ReadVariableOpReadVariableOp,read_57_disablecopyonread_false_negatives_17^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:╚*
dtype0m
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:╚d
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes	
:╚А
Read_58/DisableCopyOnReadDisableCopyOnRead+read_58_disablecopyonread_true_positives_31"/device:CPU:0*
_output_shapes
 й
Read_58/ReadVariableOpReadVariableOp+read_58_disablecopyonread_true_positives_31^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes
:Б
Read_59/DisableCopyOnReadDisableCopyOnRead,read_59_disablecopyonread_false_negatives_16"/device:CPU:0*
_output_shapes
 к
Read_59/ReadVariableOpReadVariableOp,read_59_disablecopyonread_false_negatives_16^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_60/DisableCopyOnReadDisableCopyOnRead+read_60_disablecopyonread_true_positives_30"/device:CPU:0*
_output_shapes
 й
Read_60/ReadVariableOpReadVariableOp+read_60_disablecopyonread_true_positives_30^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes
:Б
Read_61/DisableCopyOnReadDisableCopyOnRead,read_61_disablecopyonread_false_positives_16"/device:CPU:0*
_output_shapes
 к
Read_61/ReadVariableOpReadVariableOp,read_61_disablecopyonread_false_positives_16^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_62/DisableCopyOnReadDisableCopyOnRead+read_62_disablecopyonread_true_positives_29"/device:CPU:0*
_output_shapes
 й
Read_62/ReadVariableOpReadVariableOp+read_62_disablecopyonread_true_positives_29^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes
:Б
Read_63/DisableCopyOnReadDisableCopyOnRead,read_63_disablecopyonread_false_negatives_15"/device:CPU:0*
_output_shapes
 к
Read_63/ReadVariableOpReadVariableOp,read_63_disablecopyonread_false_negatives_15^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_64/DisableCopyOnReadDisableCopyOnRead+read_64_disablecopyonread_true_positives_28"/device:CPU:0*
_output_shapes
 й
Read_64/ReadVariableOpReadVariableOp+read_64_disablecopyonread_true_positives_28^Read_64/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*
_output_shapes
:Б
Read_65/DisableCopyOnReadDisableCopyOnRead,read_65_disablecopyonread_false_positives_15"/device:CPU:0*
_output_shapes
 к
Read_65/ReadVariableOpReadVariableOp,read_65_disablecopyonread_false_positives_15^Read_65/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_66/DisableCopyOnReadDisableCopyOnRead+read_66_disablecopyonread_true_positives_27"/device:CPU:0*
_output_shapes
 й
Read_66/ReadVariableOpReadVariableOp+read_66_disablecopyonread_true_positives_27^Read_66/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*
_output_shapes
:Б
Read_67/DisableCopyOnReadDisableCopyOnRead,read_67_disablecopyonread_false_negatives_14"/device:CPU:0*
_output_shapes
 к
Read_67/ReadVariableOpReadVariableOp,read_67_disablecopyonread_false_negatives_14^Read_67/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_68/DisableCopyOnReadDisableCopyOnRead+read_68_disablecopyonread_true_positives_26"/device:CPU:0*
_output_shapes
 й
Read_68/ReadVariableOpReadVariableOp+read_68_disablecopyonread_true_positives_26^Read_68/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*
_output_shapes
:Б
Read_69/DisableCopyOnReadDisableCopyOnRead,read_69_disablecopyonread_false_positives_14"/device:CPU:0*
_output_shapes
 к
Read_69/ReadVariableOpReadVariableOp,read_69_disablecopyonread_false_positives_14^Read_69/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_70/DisableCopyOnReadDisableCopyOnRead+read_70_disablecopyonread_true_positives_25"/device:CPU:0*
_output_shapes
 й
Read_70/ReadVariableOpReadVariableOp+read_70_disablecopyonread_true_positives_25^Read_70/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_140IdentityRead_70/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*
_output_shapes
:Б
Read_71/DisableCopyOnReadDisableCopyOnRead,read_71_disablecopyonread_false_negatives_13"/device:CPU:0*
_output_shapes
 к
Read_71/ReadVariableOpReadVariableOp,read_71_disablecopyonread_false_negatives_13^Read_71/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_142IdentityRead_71/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_72/DisableCopyOnReadDisableCopyOnRead+read_72_disablecopyonread_true_positives_24"/device:CPU:0*
_output_shapes
 й
Read_72/ReadVariableOpReadVariableOp+read_72_disablecopyonread_true_positives_24^Read_72/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_144IdentityRead_72/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0*
_output_shapes
:Б
Read_73/DisableCopyOnReadDisableCopyOnRead,read_73_disablecopyonread_false_positives_13"/device:CPU:0*
_output_shapes
 к
Read_73/ReadVariableOpReadVariableOp,read_73_disablecopyonread_false_positives_13^Read_73/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_146IdentityRead_73/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_74/DisableCopyOnReadDisableCopyOnRead+read_74_disablecopyonread_true_positives_23"/device:CPU:0*
_output_shapes
 й
Read_74/ReadVariableOpReadVariableOp+read_74_disablecopyonread_true_positives_23^Read_74/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_148IdentityRead_74/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_149IdentityIdentity_148:output:0"/device:CPU:0*
T0*
_output_shapes
:Б
Read_75/DisableCopyOnReadDisableCopyOnRead,read_75_disablecopyonread_false_negatives_12"/device:CPU:0*
_output_shapes
 к
Read_75/ReadVariableOpReadVariableOp,read_75_disablecopyonread_false_negatives_12^Read_75/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_150IdentityRead_75/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_151IdentityIdentity_150:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_76/DisableCopyOnReadDisableCopyOnRead+read_76_disablecopyonread_true_positives_22"/device:CPU:0*
_output_shapes
 й
Read_76/ReadVariableOpReadVariableOp+read_76_disablecopyonread_true_positives_22^Read_76/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_152IdentityRead_76/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_153IdentityIdentity_152:output:0"/device:CPU:0*
T0*
_output_shapes
:Б
Read_77/DisableCopyOnReadDisableCopyOnRead,read_77_disablecopyonread_false_positives_12"/device:CPU:0*
_output_shapes
 к
Read_77/ReadVariableOpReadVariableOp,read_77_disablecopyonread_false_positives_12^Read_77/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_154IdentityRead_77/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_155IdentityIdentity_154:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_78/DisableCopyOnReadDisableCopyOnRead+read_78_disablecopyonread_true_positives_21"/device:CPU:0*
_output_shapes
 й
Read_78/ReadVariableOpReadVariableOp+read_78_disablecopyonread_true_positives_21^Read_78/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_156IdentityRead_78/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_157IdentityIdentity_156:output:0"/device:CPU:0*
T0*
_output_shapes
:Б
Read_79/DisableCopyOnReadDisableCopyOnRead,read_79_disablecopyonread_false_negatives_11"/device:CPU:0*
_output_shapes
 к
Read_79/ReadVariableOpReadVariableOp,read_79_disablecopyonread_false_negatives_11^Read_79/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_158IdentityRead_79/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_159IdentityIdentity_158:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_80/DisableCopyOnReadDisableCopyOnRead+read_80_disablecopyonread_true_positives_20"/device:CPU:0*
_output_shapes
 й
Read_80/ReadVariableOpReadVariableOp+read_80_disablecopyonread_true_positives_20^Read_80/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_160IdentityRead_80/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_161IdentityIdentity_160:output:0"/device:CPU:0*
T0*
_output_shapes
:Б
Read_81/DisableCopyOnReadDisableCopyOnRead,read_81_disablecopyonread_false_positives_11"/device:CPU:0*
_output_shapes
 к
Read_81/ReadVariableOpReadVariableOp,read_81_disablecopyonread_false_positives_11^Read_81/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_162IdentityRead_81/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_163IdentityIdentity_162:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_82/DisableCopyOnReadDisableCopyOnRead+read_82_disablecopyonread_true_positives_19"/device:CPU:0*
_output_shapes
 й
Read_82/ReadVariableOpReadVariableOp+read_82_disablecopyonread_true_positives_19^Read_82/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_164IdentityRead_82/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_165IdentityIdentity_164:output:0"/device:CPU:0*
T0*
_output_shapes
:Б
Read_83/DisableCopyOnReadDisableCopyOnRead,read_83_disablecopyonread_false_negatives_10"/device:CPU:0*
_output_shapes
 к
Read_83/ReadVariableOpReadVariableOp,read_83_disablecopyonread_false_negatives_10^Read_83/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_166IdentityRead_83/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_167IdentityIdentity_166:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_84/DisableCopyOnReadDisableCopyOnRead+read_84_disablecopyonread_true_positives_18"/device:CPU:0*
_output_shapes
 й
Read_84/ReadVariableOpReadVariableOp+read_84_disablecopyonread_true_positives_18^Read_84/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_168IdentityRead_84/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_169IdentityIdentity_168:output:0"/device:CPU:0*
T0*
_output_shapes
:Б
Read_85/DisableCopyOnReadDisableCopyOnRead,read_85_disablecopyonread_false_positives_10"/device:CPU:0*
_output_shapes
 к
Read_85/ReadVariableOpReadVariableOp,read_85_disablecopyonread_false_positives_10^Read_85/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_170IdentityRead_85/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_171IdentityIdentity_170:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_86/DisableCopyOnReadDisableCopyOnRead+read_86_disablecopyonread_true_positives_17"/device:CPU:0*
_output_shapes
 й
Read_86/ReadVariableOpReadVariableOp+read_86_disablecopyonread_true_positives_17^Read_86/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_172IdentityRead_86/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_173IdentityIdentity_172:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_87/DisableCopyOnReadDisableCopyOnRead+read_87_disablecopyonread_false_negatives_9"/device:CPU:0*
_output_shapes
 й
Read_87/ReadVariableOpReadVariableOp+read_87_disablecopyonread_false_negatives_9^Read_87/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_174IdentityRead_87/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_175IdentityIdentity_174:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_88/DisableCopyOnReadDisableCopyOnRead+read_88_disablecopyonread_true_positives_16"/device:CPU:0*
_output_shapes
 й
Read_88/ReadVariableOpReadVariableOp+read_88_disablecopyonread_true_positives_16^Read_88/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_176IdentityRead_88/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_177IdentityIdentity_176:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_89/DisableCopyOnReadDisableCopyOnRead+read_89_disablecopyonread_false_positives_9"/device:CPU:0*
_output_shapes
 й
Read_89/ReadVariableOpReadVariableOp+read_89_disablecopyonread_false_positives_9^Read_89/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_178IdentityRead_89/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_179IdentityIdentity_178:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_90/DisableCopyOnReadDisableCopyOnRead+read_90_disablecopyonread_true_positives_15"/device:CPU:0*
_output_shapes
 й
Read_90/ReadVariableOpReadVariableOp+read_90_disablecopyonread_true_positives_15^Read_90/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_180IdentityRead_90/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_181IdentityIdentity_180:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_91/DisableCopyOnReadDisableCopyOnRead+read_91_disablecopyonread_false_negatives_8"/device:CPU:0*
_output_shapes
 й
Read_91/ReadVariableOpReadVariableOp+read_91_disablecopyonread_false_negatives_8^Read_91/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_182IdentityRead_91/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_183IdentityIdentity_182:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_92/DisableCopyOnReadDisableCopyOnRead+read_92_disablecopyonread_true_positives_14"/device:CPU:0*
_output_shapes
 й
Read_92/ReadVariableOpReadVariableOp+read_92_disablecopyonread_true_positives_14^Read_92/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_184IdentityRead_92/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_185IdentityIdentity_184:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_93/DisableCopyOnReadDisableCopyOnRead+read_93_disablecopyonread_false_positives_8"/device:CPU:0*
_output_shapes
 й
Read_93/ReadVariableOpReadVariableOp+read_93_disablecopyonread_false_positives_8^Read_93/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_186IdentityRead_93/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_187IdentityIdentity_186:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_94/DisableCopyOnReadDisableCopyOnRead+read_94_disablecopyonread_true_positives_13"/device:CPU:0*
_output_shapes
 й
Read_94/ReadVariableOpReadVariableOp+read_94_disablecopyonread_true_positives_13^Read_94/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_188IdentityRead_94/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_189IdentityIdentity_188:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_95/DisableCopyOnReadDisableCopyOnRead+read_95_disablecopyonread_false_negatives_7"/device:CPU:0*
_output_shapes
 й
Read_95/ReadVariableOpReadVariableOp+read_95_disablecopyonread_false_negatives_7^Read_95/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_190IdentityRead_95/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_191IdentityIdentity_190:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_96/DisableCopyOnReadDisableCopyOnRead+read_96_disablecopyonread_true_positives_12"/device:CPU:0*
_output_shapes
 й
Read_96/ReadVariableOpReadVariableOp+read_96_disablecopyonread_true_positives_12^Read_96/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_192IdentityRead_96/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_193IdentityIdentity_192:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_97/DisableCopyOnReadDisableCopyOnRead+read_97_disablecopyonread_false_positives_7"/device:CPU:0*
_output_shapes
 й
Read_97/ReadVariableOpReadVariableOp+read_97_disablecopyonread_false_positives_7^Read_97/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_194IdentityRead_97/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_195IdentityIdentity_194:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_98/DisableCopyOnReadDisableCopyOnRead+read_98_disablecopyonread_true_positives_11"/device:CPU:0*
_output_shapes
 й
Read_98/ReadVariableOpReadVariableOp+read_98_disablecopyonread_true_positives_11^Read_98/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_196IdentityRead_98/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_197IdentityIdentity_196:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_99/DisableCopyOnReadDisableCopyOnRead+read_99_disablecopyonread_false_negatives_6"/device:CPU:0*
_output_shapes
 й
Read_99/ReadVariableOpReadVariableOp+read_99_disablecopyonread_false_negatives_6^Read_99/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_198IdentityRead_99/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_199IdentityIdentity_198:output:0"/device:CPU:0*
T0*
_output_shapes
:В
Read_100/DisableCopyOnReadDisableCopyOnRead,read_100_disablecopyonread_true_positives_10"/device:CPU:0*
_output_shapes
 м
Read_100/ReadVariableOpReadVariableOp,read_100_disablecopyonread_true_positives_10^Read_100/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_200IdentityRead_100/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_201IdentityIdentity_200:output:0"/device:CPU:0*
T0*
_output_shapes
:В
Read_101/DisableCopyOnReadDisableCopyOnRead,read_101_disablecopyonread_false_positives_6"/device:CPU:0*
_output_shapes
 м
Read_101/ReadVariableOpReadVariableOp,read_101_disablecopyonread_false_positives_6^Read_101/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_202IdentityRead_101/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_203IdentityIdentity_202:output:0"/device:CPU:0*
T0*
_output_shapes
:Б
Read_102/DisableCopyOnReadDisableCopyOnRead+read_102_disablecopyonread_true_positives_9"/device:CPU:0*
_output_shapes
 л
Read_102/ReadVariableOpReadVariableOp+read_102_disablecopyonread_true_positives_9^Read_102/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_204IdentityRead_102/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_205IdentityIdentity_204:output:0"/device:CPU:0*
T0*
_output_shapes
:В
Read_103/DisableCopyOnReadDisableCopyOnRead,read_103_disablecopyonread_false_negatives_5"/device:CPU:0*
_output_shapes
 м
Read_103/ReadVariableOpReadVariableOp,read_103_disablecopyonread_false_negatives_5^Read_103/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_206IdentityRead_103/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_207IdentityIdentity_206:output:0"/device:CPU:0*
T0*
_output_shapes
:Б
Read_104/DisableCopyOnReadDisableCopyOnRead+read_104_disablecopyonread_true_positives_8"/device:CPU:0*
_output_shapes
 л
Read_104/ReadVariableOpReadVariableOp+read_104_disablecopyonread_true_positives_8^Read_104/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_208IdentityRead_104/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_209IdentityIdentity_208:output:0"/device:CPU:0*
T0*
_output_shapes
:В
Read_105/DisableCopyOnReadDisableCopyOnRead,read_105_disablecopyonread_false_positives_5"/device:CPU:0*
_output_shapes
 м
Read_105/ReadVariableOpReadVariableOp,read_105_disablecopyonread_false_positives_5^Read_105/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_210IdentityRead_105/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_211IdentityIdentity_210:output:0"/device:CPU:0*
T0*
_output_shapes
:Б
Read_106/DisableCopyOnReadDisableCopyOnRead+read_106_disablecopyonread_true_positives_7"/device:CPU:0*
_output_shapes
 л
Read_106/ReadVariableOpReadVariableOp+read_106_disablecopyonread_true_positives_7^Read_106/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_212IdentityRead_106/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_213IdentityIdentity_212:output:0"/device:CPU:0*
T0*
_output_shapes
:В
Read_107/DisableCopyOnReadDisableCopyOnRead,read_107_disablecopyonread_false_negatives_4"/device:CPU:0*
_output_shapes
 м
Read_107/ReadVariableOpReadVariableOp,read_107_disablecopyonread_false_negatives_4^Read_107/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_214IdentityRead_107/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_215IdentityIdentity_214:output:0"/device:CPU:0*
T0*
_output_shapes
:Б
Read_108/DisableCopyOnReadDisableCopyOnRead+read_108_disablecopyonread_true_positives_6"/device:CPU:0*
_output_shapes
 л
Read_108/ReadVariableOpReadVariableOp+read_108_disablecopyonread_true_positives_6^Read_108/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_216IdentityRead_108/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_217IdentityIdentity_216:output:0"/device:CPU:0*
T0*
_output_shapes
:В
Read_109/DisableCopyOnReadDisableCopyOnRead,read_109_disablecopyonread_false_positives_4"/device:CPU:0*
_output_shapes
 м
Read_109/ReadVariableOpReadVariableOp,read_109_disablecopyonread_false_positives_4^Read_109/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_218IdentityRead_109/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_219IdentityIdentity_218:output:0"/device:CPU:0*
T0*
_output_shapes
:Б
Read_110/DisableCopyOnReadDisableCopyOnRead+read_110_disablecopyonread_true_positives_5"/device:CPU:0*
_output_shapes
 л
Read_110/ReadVariableOpReadVariableOp+read_110_disablecopyonread_true_positives_5^Read_110/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_220IdentityRead_110/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_221IdentityIdentity_220:output:0"/device:CPU:0*
T0*
_output_shapes
:В
Read_111/DisableCopyOnReadDisableCopyOnRead,read_111_disablecopyonread_false_negatives_3"/device:CPU:0*
_output_shapes
 м
Read_111/ReadVariableOpReadVariableOp,read_111_disablecopyonread_false_negatives_3^Read_111/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_222IdentityRead_111/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_223IdentityIdentity_222:output:0"/device:CPU:0*
T0*
_output_shapes
:Б
Read_112/DisableCopyOnReadDisableCopyOnRead+read_112_disablecopyonread_true_positives_4"/device:CPU:0*
_output_shapes
 л
Read_112/ReadVariableOpReadVariableOp+read_112_disablecopyonread_true_positives_4^Read_112/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_224IdentityRead_112/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_225IdentityIdentity_224:output:0"/device:CPU:0*
T0*
_output_shapes
:В
Read_113/DisableCopyOnReadDisableCopyOnRead,read_113_disablecopyonread_false_positives_3"/device:CPU:0*
_output_shapes
 м
Read_113/ReadVariableOpReadVariableOp,read_113_disablecopyonread_false_positives_3^Read_113/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_226IdentityRead_113/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_227IdentityIdentity_226:output:0"/device:CPU:0*
T0*
_output_shapes
:Б
Read_114/DisableCopyOnReadDisableCopyOnRead+read_114_disablecopyonread_true_positives_3"/device:CPU:0*
_output_shapes
 л
Read_114/ReadVariableOpReadVariableOp+read_114_disablecopyonread_true_positives_3^Read_114/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_228IdentityRead_114/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_229IdentityIdentity_228:output:0"/device:CPU:0*
T0*
_output_shapes
:В
Read_115/DisableCopyOnReadDisableCopyOnRead,read_115_disablecopyonread_false_negatives_2"/device:CPU:0*
_output_shapes
 м
Read_115/ReadVariableOpReadVariableOp,read_115_disablecopyonread_false_negatives_2^Read_115/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_230IdentityRead_115/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_231IdentityIdentity_230:output:0"/device:CPU:0*
T0*
_output_shapes
:Б
Read_116/DisableCopyOnReadDisableCopyOnRead+read_116_disablecopyonread_true_positives_2"/device:CPU:0*
_output_shapes
 л
Read_116/ReadVariableOpReadVariableOp+read_116_disablecopyonread_true_positives_2^Read_116/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_232IdentityRead_116/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_233IdentityIdentity_232:output:0"/device:CPU:0*
T0*
_output_shapes
:В
Read_117/DisableCopyOnReadDisableCopyOnRead,read_117_disablecopyonread_false_positives_2"/device:CPU:0*
_output_shapes
 м
Read_117/ReadVariableOpReadVariableOp,read_117_disablecopyonread_false_positives_2^Read_117/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_234IdentityRead_117/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_235IdentityIdentity_234:output:0"/device:CPU:0*
T0*
_output_shapes
:Б
Read_118/DisableCopyOnReadDisableCopyOnRead+read_118_disablecopyonread_true_positives_1"/device:CPU:0*
_output_shapes
 л
Read_118/ReadVariableOpReadVariableOp+read_118_disablecopyonread_true_positives_1^Read_118/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_236IdentityRead_118/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_237IdentityIdentity_236:output:0"/device:CPU:0*
T0*
_output_shapes
:В
Read_119/DisableCopyOnReadDisableCopyOnRead,read_119_disablecopyonread_false_positives_1"/device:CPU:0*
_output_shapes
 м
Read_119/ReadVariableOpReadVariableOp,read_119_disablecopyonread_false_positives_1^Read_119/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_238IdentityRead_119/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_239IdentityIdentity_238:output:0"/device:CPU:0*
T0*
_output_shapes
:В
Read_120/DisableCopyOnReadDisableCopyOnRead,read_120_disablecopyonread_false_negatives_1"/device:CPU:0*
_output_shapes
 м
Read_120/ReadVariableOpReadVariableOp,read_120_disablecopyonread_false_negatives_1^Read_120/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_240IdentityRead_120/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_241IdentityIdentity_240:output:0"/device:CPU:0*
T0*
_output_shapes
:З
Read_121/DisableCopyOnReadDisableCopyOnRead1read_121_disablecopyonread_weights_intermediate_1"/device:CPU:0*
_output_shapes
 ▒
Read_121/ReadVariableOpReadVariableOp1read_121_disablecopyonread_weights_intermediate_1^Read_121/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_242IdentityRead_121/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_243IdentityIdentity_242:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_122/DisableCopyOnReadDisableCopyOnRead)read_122_disablecopyonread_true_positives"/device:CPU:0*
_output_shapes
 й
Read_122/ReadVariableOpReadVariableOp)read_122_disablecopyonread_true_positives^Read_122/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_244IdentityRead_122/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_245IdentityIdentity_244:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_123/DisableCopyOnReadDisableCopyOnRead*read_123_disablecopyonread_false_positives"/device:CPU:0*
_output_shapes
 к
Read_123/ReadVariableOpReadVariableOp*read_123_disablecopyonread_false_positives^Read_123/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_246IdentityRead_123/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_247IdentityIdentity_246:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_124/DisableCopyOnReadDisableCopyOnRead*read_124_disablecopyonread_false_negatives"/device:CPU:0*
_output_shapes
 к
Read_124/ReadVariableOpReadVariableOp*read_124_disablecopyonread_false_negatives^Read_124/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_248IdentityRead_124/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_249IdentityIdentity_248:output:0"/device:CPU:0*
T0*
_output_shapes
:Е
Read_125/DisableCopyOnReadDisableCopyOnRead/read_125_disablecopyonread_weights_intermediate"/device:CPU:0*
_output_shapes
 п
Read_125/ReadVariableOpReadVariableOp/read_125_disablecopyonread_weights_intermediate^Read_125/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_250IdentityRead_125/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_251IdentityIdentity_250:output:0"/device:CPU:0*
T0*
_output_shapes
:ь9
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Х9
valueЛ9BИ9B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB;optimizer/_current_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/5/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/5/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/6/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/6/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/7/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/7/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/8/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/8/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/9/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/9/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/10/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/10/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/11/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/11/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/12/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/12/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/13/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/13/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/14/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/14/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/15/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/15/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/16/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/16/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/17/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/17/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/18/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/18/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/19/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/19/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/20/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/20/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/21/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/21/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/22/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/22/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/23/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/23/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/24/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/24/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/25/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/25/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/26/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/26/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/27/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/27/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/28/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/28/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/29/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/29/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/30/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/30/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/31/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/31/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/32/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/32/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/33/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/33/false_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/33/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBDkeras_api/metrics/33/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/34/true_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/34/false_positives/.ATTRIBUTES/VARIABLE_VALUEB?keras_api/metrics/34/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBDkeras_api/metrics/34/weights_intermediate/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHю
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*У
valueЙBЖB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ї
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0Identity_149:output:0Identity_151:output:0Identity_153:output:0Identity_155:output:0Identity_157:output:0Identity_159:output:0Identity_161:output:0Identity_163:output:0Identity_165:output:0Identity_167:output:0Identity_169:output:0Identity_171:output:0Identity_173:output:0Identity_175:output:0Identity_177:output:0Identity_179:output:0Identity_181:output:0Identity_183:output:0Identity_185:output:0Identity_187:output:0Identity_189:output:0Identity_191:output:0Identity_193:output:0Identity_195:output:0Identity_197:output:0Identity_199:output:0Identity_201:output:0Identity_203:output:0Identity_205:output:0Identity_207:output:0Identity_209:output:0Identity_211:output:0Identity_213:output:0Identity_215:output:0Identity_217:output:0Identity_219:output:0Identity_221:output:0Identity_223:output:0Identity_225:output:0Identity_227:output:0Identity_229:output:0Identity_231:output:0Identity_233:output:0Identity_235:output:0Identity_237:output:0Identity_239:output:0Identity_241:output:0Identity_243:output:0Identity_245:output:0Identity_247:output:0Identity_249:output:0Identity_251:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *П
dtypesД
Б2	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_252Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_253IdentityIdentity_252:output:0^NoOp*
T0*
_output_shapes
: щ4
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_100/DisableCopyOnRead^Read_100/ReadVariableOp^Read_101/DisableCopyOnRead^Read_101/ReadVariableOp^Read_102/DisableCopyOnRead^Read_102/ReadVariableOp^Read_103/DisableCopyOnRead^Read_103/ReadVariableOp^Read_104/DisableCopyOnRead^Read_104/ReadVariableOp^Read_105/DisableCopyOnRead^Read_105/ReadVariableOp^Read_106/DisableCopyOnRead^Read_106/ReadVariableOp^Read_107/DisableCopyOnRead^Read_107/ReadVariableOp^Read_108/DisableCopyOnRead^Read_108/ReadVariableOp^Read_109/DisableCopyOnRead^Read_109/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_110/DisableCopyOnRead^Read_110/ReadVariableOp^Read_111/DisableCopyOnRead^Read_111/ReadVariableOp^Read_112/DisableCopyOnRead^Read_112/ReadVariableOp^Read_113/DisableCopyOnRead^Read_113/ReadVariableOp^Read_114/DisableCopyOnRead^Read_114/ReadVariableOp^Read_115/DisableCopyOnRead^Read_115/ReadVariableOp^Read_116/DisableCopyOnRead^Read_116/ReadVariableOp^Read_117/DisableCopyOnRead^Read_117/ReadVariableOp^Read_118/DisableCopyOnRead^Read_118/ReadVariableOp^Read_119/DisableCopyOnRead^Read_119/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_120/DisableCopyOnRead^Read_120/ReadVariableOp^Read_121/DisableCopyOnRead^Read_121/ReadVariableOp^Read_122/DisableCopyOnRead^Read_122/ReadVariableOp^Read_123/DisableCopyOnRead^Read_123/ReadVariableOp^Read_124/DisableCopyOnRead^Read_124/ReadVariableOp^Read_125/DisableCopyOnRead^Read_125/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_74/DisableCopyOnRead^Read_74/ReadVariableOp^Read_75/DisableCopyOnRead^Read_75/ReadVariableOp^Read_76/DisableCopyOnRead^Read_76/ReadVariableOp^Read_77/DisableCopyOnRead^Read_77/ReadVariableOp^Read_78/DisableCopyOnRead^Read_78/ReadVariableOp^Read_79/DisableCopyOnRead^Read_79/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_80/DisableCopyOnRead^Read_80/ReadVariableOp^Read_81/DisableCopyOnRead^Read_81/ReadVariableOp^Read_82/DisableCopyOnRead^Read_82/ReadVariableOp^Read_83/DisableCopyOnRead^Read_83/ReadVariableOp^Read_84/DisableCopyOnRead^Read_84/ReadVariableOp^Read_85/DisableCopyOnRead^Read_85/ReadVariableOp^Read_86/DisableCopyOnRead^Read_86/ReadVariableOp^Read_87/DisableCopyOnRead^Read_87/ReadVariableOp^Read_88/DisableCopyOnRead^Read_88/ReadVariableOp^Read_89/DisableCopyOnRead^Read_89/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp^Read_90/DisableCopyOnRead^Read_90/ReadVariableOp^Read_91/DisableCopyOnRead^Read_91/ReadVariableOp^Read_92/DisableCopyOnRead^Read_92/ReadVariableOp^Read_93/DisableCopyOnRead^Read_93/ReadVariableOp^Read_94/DisableCopyOnRead^Read_94/ReadVariableOp^Read_95/DisableCopyOnRead^Read_95/ReadVariableOp^Read_96/DisableCopyOnRead^Read_96/ReadVariableOp^Read_97/DisableCopyOnRead^Read_97/ReadVariableOp^Read_98/DisableCopyOnRead^Read_98/ReadVariableOp^Read_99/DisableCopyOnRead^Read_99/ReadVariableOp*
_output_shapes
 "%
identity_253Identity_253:output:0*(
_construction_contextkEagerRuntime*Х
_input_shapesГ
А: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp28
Read_100/DisableCopyOnReadRead_100/DisableCopyOnRead22
Read_100/ReadVariableOpRead_100/ReadVariableOp28
Read_101/DisableCopyOnReadRead_101/DisableCopyOnRead22
Read_101/ReadVariableOpRead_101/ReadVariableOp28
Read_102/DisableCopyOnReadRead_102/DisableCopyOnRead22
Read_102/ReadVariableOpRead_102/ReadVariableOp28
Read_103/DisableCopyOnReadRead_103/DisableCopyOnRead22
Read_103/ReadVariableOpRead_103/ReadVariableOp28
Read_104/DisableCopyOnReadRead_104/DisableCopyOnRead22
Read_104/ReadVariableOpRead_104/ReadVariableOp28
Read_105/DisableCopyOnReadRead_105/DisableCopyOnRead22
Read_105/ReadVariableOpRead_105/ReadVariableOp28
Read_106/DisableCopyOnReadRead_106/DisableCopyOnRead22
Read_106/ReadVariableOpRead_106/ReadVariableOp28
Read_107/DisableCopyOnReadRead_107/DisableCopyOnRead22
Read_107/ReadVariableOpRead_107/ReadVariableOp28
Read_108/DisableCopyOnReadRead_108/DisableCopyOnRead22
Read_108/ReadVariableOpRead_108/ReadVariableOp28
Read_109/DisableCopyOnReadRead_109/DisableCopyOnRead22
Read_109/ReadVariableOpRead_109/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp28
Read_110/DisableCopyOnReadRead_110/DisableCopyOnRead22
Read_110/ReadVariableOpRead_110/ReadVariableOp28
Read_111/DisableCopyOnReadRead_111/DisableCopyOnRead22
Read_111/ReadVariableOpRead_111/ReadVariableOp28
Read_112/DisableCopyOnReadRead_112/DisableCopyOnRead22
Read_112/ReadVariableOpRead_112/ReadVariableOp28
Read_113/DisableCopyOnReadRead_113/DisableCopyOnRead22
Read_113/ReadVariableOpRead_113/ReadVariableOp28
Read_114/DisableCopyOnReadRead_114/DisableCopyOnRead22
Read_114/ReadVariableOpRead_114/ReadVariableOp28
Read_115/DisableCopyOnReadRead_115/DisableCopyOnRead22
Read_115/ReadVariableOpRead_115/ReadVariableOp28
Read_116/DisableCopyOnReadRead_116/DisableCopyOnRead22
Read_116/ReadVariableOpRead_116/ReadVariableOp28
Read_117/DisableCopyOnReadRead_117/DisableCopyOnRead22
Read_117/ReadVariableOpRead_117/ReadVariableOp28
Read_118/DisableCopyOnReadRead_118/DisableCopyOnRead22
Read_118/ReadVariableOpRead_118/ReadVariableOp28
Read_119/DisableCopyOnReadRead_119/DisableCopyOnRead22
Read_119/ReadVariableOpRead_119/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp28
Read_120/DisableCopyOnReadRead_120/DisableCopyOnRead22
Read_120/ReadVariableOpRead_120/ReadVariableOp28
Read_121/DisableCopyOnReadRead_121/DisableCopyOnRead22
Read_121/ReadVariableOpRead_121/ReadVariableOp28
Read_122/DisableCopyOnReadRead_122/DisableCopyOnRead22
Read_122/ReadVariableOpRead_122/ReadVariableOp28
Read_123/DisableCopyOnReadRead_123/DisableCopyOnRead22
Read_123/ReadVariableOpRead_123/ReadVariableOp28
Read_124/DisableCopyOnReadRead_124/DisableCopyOnRead22
Read_124/ReadVariableOpRead_124/ReadVariableOp28
Read_125/DisableCopyOnReadRead_125/DisableCopyOnRead22
Read_125/ReadVariableOpRead_125/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp26
Read_74/DisableCopyOnReadRead_74/DisableCopyOnRead20
Read_74/ReadVariableOpRead_74/ReadVariableOp26
Read_75/DisableCopyOnReadRead_75/DisableCopyOnRead20
Read_75/ReadVariableOpRead_75/ReadVariableOp26
Read_76/DisableCopyOnReadRead_76/DisableCopyOnRead20
Read_76/ReadVariableOpRead_76/ReadVariableOp26
Read_77/DisableCopyOnReadRead_77/DisableCopyOnRead20
Read_77/ReadVariableOpRead_77/ReadVariableOp26
Read_78/DisableCopyOnReadRead_78/DisableCopyOnRead20
Read_78/ReadVariableOpRead_78/ReadVariableOp26
Read_79/DisableCopyOnReadRead_79/DisableCopyOnRead20
Read_79/ReadVariableOpRead_79/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp26
Read_80/DisableCopyOnReadRead_80/DisableCopyOnRead20
Read_80/ReadVariableOpRead_80/ReadVariableOp26
Read_81/DisableCopyOnReadRead_81/DisableCopyOnRead20
Read_81/ReadVariableOpRead_81/ReadVariableOp26
Read_82/DisableCopyOnReadRead_82/DisableCopyOnRead20
Read_82/ReadVariableOpRead_82/ReadVariableOp26
Read_83/DisableCopyOnReadRead_83/DisableCopyOnRead20
Read_83/ReadVariableOpRead_83/ReadVariableOp26
Read_84/DisableCopyOnReadRead_84/DisableCopyOnRead20
Read_84/ReadVariableOpRead_84/ReadVariableOp26
Read_85/DisableCopyOnReadRead_85/DisableCopyOnRead20
Read_85/ReadVariableOpRead_85/ReadVariableOp26
Read_86/DisableCopyOnReadRead_86/DisableCopyOnRead20
Read_86/ReadVariableOpRead_86/ReadVariableOp26
Read_87/DisableCopyOnReadRead_87/DisableCopyOnRead20
Read_87/ReadVariableOpRead_87/ReadVariableOp26
Read_88/DisableCopyOnReadRead_88/DisableCopyOnRead20
Read_88/ReadVariableOpRead_88/ReadVariableOp26
Read_89/DisableCopyOnReadRead_89/DisableCopyOnRead20
Read_89/ReadVariableOpRead_89/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp26
Read_90/DisableCopyOnReadRead_90/DisableCopyOnRead20
Read_90/ReadVariableOpRead_90/ReadVariableOp26
Read_91/DisableCopyOnReadRead_91/DisableCopyOnRead20
Read_91/ReadVariableOpRead_91/ReadVariableOp26
Read_92/DisableCopyOnReadRead_92/DisableCopyOnRead20
Read_92/ReadVariableOpRead_92/ReadVariableOp26
Read_93/DisableCopyOnReadRead_93/DisableCopyOnRead20
Read_93/ReadVariableOpRead_93/ReadVariableOp26
Read_94/DisableCopyOnReadRead_94/DisableCopyOnRead20
Read_94/ReadVariableOpRead_94/ReadVariableOp26
Read_95/DisableCopyOnReadRead_95/DisableCopyOnRead20
Read_95/ReadVariableOpRead_95/ReadVariableOp26
Read_96/DisableCopyOnReadRead_96/DisableCopyOnRead20
Read_96/ReadVariableOpRead_96/ReadVariableOp26
Read_97/DisableCopyOnReadRead_97/DisableCopyOnRead20
Read_97/ReadVariableOpRead_97/ReadVariableOp26
Read_98/DisableCopyOnReadRead_98/DisableCopyOnRead20
Read_98/ReadVariableOpRead_98/ReadVariableOp26
Read_99/DisableCopyOnReadRead_99/DisableCopyOnRead20
Read_99/ReadVariableOpRead_99/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:4~0
.
_user_specified_nameweights_intermediate:/}+
)
_user_specified_namefalse_negatives:/|+
)
_user_specified_namefalse_positives:.{*
(
_user_specified_nametrue_positives:6z2
0
_user_specified_nameweights_intermediate_1:1y-
+
_user_specified_namefalse_negatives_1:1x-
+
_user_specified_namefalse_positives_1:0w,
*
_user_specified_nametrue_positives_1:1v-
+
_user_specified_namefalse_positives_2:0u,
*
_user_specified_nametrue_positives_2:1t-
+
_user_specified_namefalse_negatives_2:0s,
*
_user_specified_nametrue_positives_3:1r-
+
_user_specified_namefalse_positives_3:0q,
*
_user_specified_nametrue_positives_4:1p-
+
_user_specified_namefalse_negatives_3:0o,
*
_user_specified_nametrue_positives_5:1n-
+
_user_specified_namefalse_positives_4:0m,
*
_user_specified_nametrue_positives_6:1l-
+
_user_specified_namefalse_negatives_4:0k,
*
_user_specified_nametrue_positives_7:1j-
+
_user_specified_namefalse_positives_5:0i,
*
_user_specified_nametrue_positives_8:1h-
+
_user_specified_namefalse_negatives_5:0g,
*
_user_specified_nametrue_positives_9:1f-
+
_user_specified_namefalse_positives_6:1e-
+
_user_specified_nametrue_positives_10:1d-
+
_user_specified_namefalse_negatives_6:1c-
+
_user_specified_nametrue_positives_11:1b-
+
_user_specified_namefalse_positives_7:1a-
+
_user_specified_nametrue_positives_12:1`-
+
_user_specified_namefalse_negatives_7:1_-
+
_user_specified_nametrue_positives_13:1^-
+
_user_specified_namefalse_positives_8:1]-
+
_user_specified_nametrue_positives_14:1\-
+
_user_specified_namefalse_negatives_8:1[-
+
_user_specified_nametrue_positives_15:1Z-
+
_user_specified_namefalse_positives_9:1Y-
+
_user_specified_nametrue_positives_16:1X-
+
_user_specified_namefalse_negatives_9:1W-
+
_user_specified_nametrue_positives_17:2V.
,
_user_specified_namefalse_positives_10:1U-
+
_user_specified_nametrue_positives_18:2T.
,
_user_specified_namefalse_negatives_10:1S-
+
_user_specified_nametrue_positives_19:2R.
,
_user_specified_namefalse_positives_11:1Q-
+
_user_specified_nametrue_positives_20:2P.
,
_user_specified_namefalse_negatives_11:1O-
+
_user_specified_nametrue_positives_21:2N.
,
_user_specified_namefalse_positives_12:1M-
+
_user_specified_nametrue_positives_22:2L.
,
_user_specified_namefalse_negatives_12:1K-
+
_user_specified_nametrue_positives_23:2J.
,
_user_specified_namefalse_positives_13:1I-
+
_user_specified_nametrue_positives_24:2H.
,
_user_specified_namefalse_negatives_13:1G-
+
_user_specified_nametrue_positives_25:2F.
,
_user_specified_namefalse_positives_14:1E-
+
_user_specified_nametrue_positives_26:2D.
,
_user_specified_namefalse_negatives_14:1C-
+
_user_specified_nametrue_positives_27:2B.
,
_user_specified_namefalse_positives_15:1A-
+
_user_specified_nametrue_positives_28:2@.
,
_user_specified_namefalse_negatives_15:1?-
+
_user_specified_nametrue_positives_29:2>.
,
_user_specified_namefalse_positives_16:1=-
+
_user_specified_nametrue_positives_30:2<.
,
_user_specified_namefalse_negatives_16:1;-
+
_user_specified_nametrue_positives_31:2:.
,
_user_specified_namefalse_negatives_17:29.
,
_user_specified_namefalse_positives_17:.8*
(
_user_specified_nametrue_negatives:17-
+
_user_specified_nametrue_positives_32:%6!

_user_specified_namecount:%5!

_user_specified_nametotal:'4#
!
_user_specified_name	count_1:'3#
!
_user_specified_name	total_1:32/
-
_user_specified_nameAdam/v/dense_4/bias:31/
-
_user_specified_nameAdam/m/dense_4/bias:501
/
_user_specified_nameAdam/v/dense_4/kernel:5/1
/
_user_specified_nameAdam/m/dense_4/kernel:3./
-
_user_specified_nameAdam/v/dense_3/bias:3-/
-
_user_specified_nameAdam/m/dense_3/bias:5,1
/
_user_specified_nameAdam/v/dense_3/kernel:5+1
/
_user_specified_nameAdam/m/dense_3/kernel:3*/
-
_user_specified_nameAdam/v/dense_2/bias:3)/
-
_user_specified_nameAdam/m/dense_2/bias:5(1
/
_user_specified_nameAdam/v/dense_2/kernel:5'1
/
_user_specified_nameAdam/m/dense_2/kernel:3&/
-
_user_specified_nameAdam/v/dense_1/bias:3%/
-
_user_specified_nameAdam/m/dense_1/bias:5$1
/
_user_specified_nameAdam/v/dense_1/kernel:5#1
/
_user_specified_nameAdam/m/dense_1/kernel:1"-
+
_user_specified_nameAdam/v/dense/bias:1!-
+
_user_specified_nameAdam/m/dense/bias:3 /
-
_user_specified_nameAdam/v/dense/kernel:3/
-
_user_specified_nameAdam/m/dense/kernel:=9
7
_user_specified_nameAdam/v/update_connection/bias:=9
7
_user_specified_nameAdam/m/update_connection/bias:IE
C
_user_specified_name+)Adam/v/update_connection/recurrent_kernel:IE
C
_user_specified_name+)Adam/m/update_connection/recurrent_kernel:?;
9
_user_specified_name!Adam/v/update_connection/kernel:?;
9
_user_specified_name!Adam/m/update_connection/kernel:51
/
_user_specified_nameAdam/v/update_ip/bias:51
/
_user_specified_nameAdam/m/update_ip/bias:A=
;
_user_specified_name#!Adam/v/update_ip/recurrent_kernel:A=
;
_user_specified_name#!Adam/m/update_ip/recurrent_kernel:73
1
_user_specified_nameAdam/v/update_ip/kernel:73
1
_user_specified_nameAdam/m/update_ip/kernel:51
/
_user_specified_namecurrent_learning_rate:)%
#
_user_specified_name	iteration:,(
&
_user_specified_namedense_4/bias:.*
(
_user_specified_namedense_4/kernel:,(
&
_user_specified_namedense_3/bias:.*
(
_user_specified_namedense_3/kernel:,(
&
_user_specified_namedense_2/bias:.*
(
_user_specified_namedense_2/kernel:,
(
&
_user_specified_namedense_1/bias:.	*
(
_user_specified_namedense_1/kernel:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:62
0
_user_specified_nameupdate_connection/bias:B>
<
_user_specified_name$"update_connection/recurrent_kernel:84
2
_user_specified_nameupdate_connection/kernel:.*
(
_user_specified_nameupdate_ip/bias::6
4
_user_specified_nameupdate_ip/recurrent_kernel:0,
*
_user_specified_nameupdate_ip/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ъ

e
F__inference_dropout_3_layer_call_and_return_conditional_losses_4845637

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         @Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧М
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ж
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    У
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*'
_output_shapes
:         @a
IdentityIdentitydropout/SelectV2:output:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Д
Щ
I__inference_sequential_2_layer_call_and_return_conditional_losses_4845687
input_3#
dense_2_4845659:
АА
dense_2_4845661:	А"
dense_3_4845670:	А@
dense_3_4845672:@!
dense_4_4845681:@
dense_4_4845683:
identityИвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallё
dense_2/StatefulPartitionedCallStatefulPartitionedCallinput_3dense_2_4845659dense_2_4845661*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_4845591▐
dropout_2/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_4845668Л
dense_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_3_4845670dense_3_4845672*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_4845620▌
dropout_3/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_4845679Л
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_4_4845681dense_4_4845683*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_4845649w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         И
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:'#
!
_user_specified_name	4845683:'#
!
_user_specified_name	4845681:'#
!
_user_specified_name	4845672:'#
!
_user_specified_name	4845670:'#
!
_user_specified_name	4845661:'#
!
_user_specified_name	4845659:Q M
(
_output_shapes
:         А
!
_user_specified_name	input_3
У
ф
N__inference_update_connection_layer_call_and_return_conditional_losses_4847668

inputs
states_0*
readvariableop_resource:	А2
matmul_readvariableop_resource:
АА4
 matmul_1_readvariableop_resource:
АА
identity

identity_1ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	А*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:А:А*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         АZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         АZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"А   А       \
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╔
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         АN
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         Аc
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         АR
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         А^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         АZ
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         АJ
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:         АV
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:         АJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         АR
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:         АW
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         АY
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         А[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         Аe
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         А:                  : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:ZV
0
_output_shapes
:                  
"
_user_specified_name
states_0:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╩╫
ж
@__inference_gnn_layer_call_and_return_conditional_losses_4846519
dst_connection_to_ip	
dst_ip_to_connection	
feature_connection
n_c	
n_i	
src_connection_to_ip	
src_ip_to_connection	&
sequential_4845802:
АА!
sequential_4845804:	А(
sequential_1_4845837:
АА#
sequential_1_4845839:	А$
update_ip_4845903:	А%
update_ip_4845905:
АА%
update_ip_4845907:
АА,
update_connection_4845951:	А-
update_connection_4845953:
АА-
update_connection_4845955:
АА(
sequential_2_4846505:
АА#
sequential_2_4846507:	А'
sequential_2_4846509:	А@"
sequential_2_4846511:@&
sequential_2_4846513:@"
sequential_2_4846515:
identityИв"sequential/StatefulPartitionedCallв$sequential/StatefulPartitionedCall_1в$sequential/StatefulPartitionedCall_2в$sequential/StatefulPartitionedCall_3в$sequential/StatefulPartitionedCall_4в$sequential/StatefulPartitionedCall_5в$sequential/StatefulPartitionedCall_6в$sequential/StatefulPartitionedCall_7в$sequential_1/StatefulPartitionedCallв&sequential_1/StatefulPartitionedCall_1в&sequential_1/StatefulPartitionedCall_2в&sequential_1/StatefulPartitionedCall_3в&sequential_1/StatefulPartitionedCall_4в&sequential_1/StatefulPartitionedCall_5в&sequential_1/StatefulPartitionedCall_6в&sequential_1/StatefulPartitionedCall_7в$sequential_2/StatefulPartitionedCallв)update_connection/StatefulPartitionedCallв+update_connection/StatefulPartitionedCall_1в+update_connection/StatefulPartitionedCall_2в+update_connection/StatefulPartitionedCall_3в+update_connection/StatefulPartitionedCall_4в+update_connection/StatefulPartitionedCall_5в+update_connection/StatefulPartitionedCall_6в+update_connection/StatefulPartitionedCall_7в!update_ip/StatefulPartitionedCallв#update_ip/StatefulPartitionedCall_1в#update_ip/StatefulPartitionedCall_2в#update_ip/StatefulPartitionedCall_3в#update_ip/StatefulPartitionedCall_4в#update_ip/StatefulPartitionedCall_5в#update_ip/StatefulPartitionedCall_6в#update_ip/StatefulPartitionedCall_7I
SqueezeSqueezefeature_connection*
T0*
_output_shapes
:M
	Squeeze_1Squeezesrc_ip_to_connection*
T0	*
_output_shapes
:M
	Squeeze_2Squeezedst_ip_to_connection*
T0	*
_output_shapes
:M
	Squeeze_3Squeezesrc_connection_to_ip*
T0	*
_output_shapes
:M
	Squeeze_4Squeezedst_connection_to_ip*
T0	*
_output_shapes
:F
	ones/CastCastn_i*

DstT0*

SrcT0	*
_output_shapes
: P
ones/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Аh
ones/packedPackones/Cast:y:0ones/packed/1:output:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?j
onesFillones/packed:output:0ones/Const:output:0*
T0*(
_output_shapes
:         АI
stack/1Const*
_output_shapes
: *
dtype0	*
value	B	 RfR
stackPackn_cstack/1:output:0*
N*
T0	*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    w
zerosFillstack:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         f*

index_type0	M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :О
concatConcatV2Squeeze:output:0zeros:output:0concat/axis:output:0*
N*
T0*0
_output_shapes
:                  O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Х
GatherV2GatherV2ones:output:0Squeeze_1:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:Q
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ы

GatherV2_1GatherV2concat:output:0Squeeze_2:output:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:L
	Squeeze_5SqueezeGatherV2_1:output:0*
T0*
_output_shapes
:J
	Squeeze_6SqueezeGatherV2:output:0*
T0*
_output_shapes
:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :А
concat_1ConcatV2Squeeze_6:output:0Squeeze_5:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:
EnsureShapeEnsureShapeconcat_1:output:0*
T0*(
_output_shapes
:         А*
shape:         АК
"sequential/StatefulPartitionedCallStatefulPartitionedCallEnsureShape:output:0sequential_4845802sequential_4845804*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4845441r
UnsortedSegmentMean/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧c
UnsortedSegmentMean/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Р
UnsortedSegmentMean/onesFill"UnsortedSegmentMean/Shape:output:0'UnsortedSegmentMean/ones/Const:output:0*
T0*
_output_shapes
:┐
&UnsortedSegmentMean/UnsortedSegmentSumUnsortedSegmentSum!UnsortedSegmentMean/ones:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:q
'UnsortedSegmentMean/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)UnsortedSegmentMean/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: s
)UnsortedSegmentMean/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ч
!UnsortedSegmentMean/strided_sliceStridedSlicen_c0UnsortedSegmentMean/strided_slice/stack:output:02UnsortedSegmentMean/strided_slice/stack_1:output:02UnsortedSegmentMean/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_maskZ
UnsortedSegmentMean/RankConst*
_output_shapes
: *
dtype0*
value	B :W
UnsortedSegmentMean/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: З
UnsortedSegmentMean/subSub!UnsortedSegmentMean/Rank:output:0#UnsortedSegmentMean/Rank_1:output:0*
T0*
_output_shapes
: t
!UnsortedSegmentMean/ones_1/packedPackUnsortedSegmentMean/sub:z:0*
N*
T0*
_output_shapes
:b
 UnsortedSegmentMean/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rз
UnsortedSegmentMean/ones_1Fill*UnsortedSegmentMean/ones_1/packed:output:0)UnsortedSegmentMean/ones_1/Const:output:0*
T0	*#
_output_shapes
:         a
UnsortedSegmentMean/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╪
UnsortedSegmentMean/concatConcatV2*UnsortedSegmentMean/strided_slice:output:0#UnsortedSegmentMean/ones_1:output:0(UnsortedSegmentMean/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         н
UnsortedSegmentMean/ReshapeReshape/UnsortedSegmentMean/UnsortedSegmentSum:output:0#UnsortedSegmentMean/concat:output:0*
Tshape0	*
T0*
_output_shapes
:b
UnsortedSegmentMean/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ч
UnsortedSegmentMean/MaximumMaximum$UnsortedSegmentMean/Reshape:output:0&UnsortedSegmentMean/Maximum/y:output:0*
T0*
_output_shapes
:╦
(UnsortedSegmentMean/UnsortedSegmentSum_1UnsortedSegmentSum+sequential/StatefulPartitionedCall:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:Э
UnsortedSegmentMean/truedivRealDiv1UnsortedSegmentMean/UnsortedSegmentSum_1:output:0UnsortedSegmentMean/Maximum:z:0*
T0*
_output_shapes
:Q
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ы

GatherV2_2GatherV2concat:output:0Squeeze_3:output:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:Q
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : Щ

GatherV2_3GatherV2ones:output:0Squeeze_4:output:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:L
	Squeeze_7SqueezeGatherV2_3:output:0*
T0*
_output_shapes
:L
	Squeeze_8SqueezeGatherV2_2:output:0*
T0*
_output_shapes
:O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :А
concat_2ConcatV2Squeeze_8:output:0Squeeze_7:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:Б
EnsureShape_1EnsureShapeconcat_2:output:0*
T0*(
_output_shapes
:         А*
shape:         АФ
$sequential_1/StatefulPartitionedCallStatefulPartitionedCallEnsureShape_1:output:0sequential_1_4845837sequential_1_4845839*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_4845526t
UnsortedSegmentMean_1/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_1/onesFill$UnsortedSegmentMean_1/Shape:output:0)UnsortedSegmentMean_1/ones/Const:output:0*
T0*
_output_shapes
:├
(UnsortedSegmentMean_1/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_1/ones:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
#UnsortedSegmentMean_1/strided_sliceStridedSlicen_i2UnsortedSegmentMean_1/strided_slice/stack:output:04UnsortedSegmentMean_1/strided_slice/stack_1:output:04UnsortedSegmentMean_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_1/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_1/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_1/subSub#UnsortedSegmentMean_1/Rank:output:0%UnsortedSegmentMean_1/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_1/ones_1/packedPackUnsortedSegmentMean_1/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_1/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_1/ones_1Fill,UnsortedSegmentMean_1/ones_1/packed:output:0+UnsortedSegmentMean_1/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_1/concatConcatV2,UnsortedSegmentMean_1/strided_slice:output:0%UnsortedSegmentMean_1/ones_1:output:0*UnsortedSegmentMean_1/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_1/ReshapeReshape1UnsortedSegmentMean_1/UnsortedSegmentSum:output:0%UnsortedSegmentMean_1/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_1/MaximumMaximum&UnsortedSegmentMean_1/Reshape:output:0(UnsortedSegmentMean_1/Maximum/y:output:0*
T0*
_output_shapes
:╧
*UnsortedSegmentMean_1/UnsortedSegmentSum_1UnsortedSegmentSum-sequential_1/StatefulPartitionedCall:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_1/truedivRealDiv3UnsortedSegmentMean_1/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_1/Maximum:z:0*
T0*
_output_shapes
:С
EnsureShape_2EnsureShape!UnsortedSegmentMean_1/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         А┬
!update_ip/StatefulPartitionedCallStatefulPartitionedCallEnsureShape_2:output:0ones:output:0update_ip_4845903update_ip_4845905update_ip_4845907*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_update_ip_layer_call_and_return_conditional_losses_4845902П
EnsureShape_3EnsureShapeUnsortedSegmentMean/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         Аь
)update_connection/StatefulPartitionedCallStatefulPartitionedCallEnsureShape_3:output:0concat:output:0update_connection_4845951update_connection_4845953update_connection_4845955*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_update_connection_layer_call_and_return_conditional_losses_4845950Q
GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B : ╢

GatherV2_4GatherV2*update_ip/StatefulPartitionedCall:output:0Squeeze_1:output:0GatherV2_4/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:Q
GatherV2_5/axisConst*
_output_shapes
: *
dtype0*
value	B : ╛

GatherV2_5GatherV22update_connection/StatefulPartitionedCall:output:0Squeeze_2:output:0GatherV2_5/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:L
	Squeeze_9SqueezeGatherV2_5:output:0*
T0*
_output_shapes
:M

Squeeze_10SqueezeGatherV2_4:output:0*
T0*
_output_shapes
:O
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B :Б
concat_3ConcatV2Squeeze_10:output:0Squeeze_9:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:Б
EnsureShape_4EnsureShapeconcat_3:output:0*
T0*(
_output_shapes
:         А*
shape:         АО
$sequential/StatefulPartitionedCall_1StatefulPartitionedCallEnsureShape_4:output:0sequential_4845802sequential_4845804*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4845441t
UnsortedSegmentMean_2/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_2/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_2/onesFill$UnsortedSegmentMean_2/Shape:output:0)UnsortedSegmentMean_2/ones/Const:output:0*
T0*
_output_shapes
:├
(UnsortedSegmentMean_2/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_2/ones:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
#UnsortedSegmentMean_2/strided_sliceStridedSlicen_c2UnsortedSegmentMean_2/strided_slice/stack:output:04UnsortedSegmentMean_2/strided_slice/stack_1:output:04UnsortedSegmentMean_2/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_2/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_2/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_2/subSub#UnsortedSegmentMean_2/Rank:output:0%UnsortedSegmentMean_2/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_2/ones_1/packedPackUnsortedSegmentMean_2/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_2/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_2/ones_1Fill,UnsortedSegmentMean_2/ones_1/packed:output:0+UnsortedSegmentMean_2/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_2/concatConcatV2,UnsortedSegmentMean_2/strided_slice:output:0%UnsortedSegmentMean_2/ones_1:output:0*UnsortedSegmentMean_2/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_2/ReshapeReshape1UnsortedSegmentMean_2/UnsortedSegmentSum:output:0%UnsortedSegmentMean_2/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_2/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_2/MaximumMaximum&UnsortedSegmentMean_2/Reshape:output:0(UnsortedSegmentMean_2/Maximum/y:output:0*
T0*
_output_shapes
:╧
*UnsortedSegmentMean_2/UnsortedSegmentSum_1UnsortedSegmentSum-sequential/StatefulPartitionedCall_1:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_2/truedivRealDiv3UnsortedSegmentMean_2/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_2/Maximum:z:0*
T0*
_output_shapes
:Q
GatherV2_6/axisConst*
_output_shapes
: *
dtype0*
value	B : ╛

GatherV2_6GatherV22update_connection/StatefulPartitionedCall:output:0Squeeze_3:output:0GatherV2_6/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:Q
GatherV2_7/axisConst*
_output_shapes
: *
dtype0*
value	B : ╢

GatherV2_7GatherV2*update_ip/StatefulPartitionedCall:output:0Squeeze_4:output:0GatherV2_7/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:M

Squeeze_11SqueezeGatherV2_7:output:0*
T0*
_output_shapes
:M

Squeeze_12SqueezeGatherV2_6:output:0*
T0*
_output_shapes
:O
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concat_4ConcatV2Squeeze_12:output:0Squeeze_11:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:Б
EnsureShape_5EnsureShapeconcat_4:output:0*
T0*(
_output_shapes
:         А*
shape:         АЦ
&sequential_1/StatefulPartitionedCall_1StatefulPartitionedCallEnsureShape_5:output:0sequential_1_4845837sequential_1_4845839*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_4845526t
UnsortedSegmentMean_3/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_3/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_3/onesFill$UnsortedSegmentMean_3/Shape:output:0)UnsortedSegmentMean_3/ones/Const:output:0*
T0*
_output_shapes
:├
(UnsortedSegmentMean_3/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_3/ones:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
#UnsortedSegmentMean_3/strided_sliceStridedSlicen_i2UnsortedSegmentMean_3/strided_slice/stack:output:04UnsortedSegmentMean_3/strided_slice/stack_1:output:04UnsortedSegmentMean_3/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_3/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_3/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_3/subSub#UnsortedSegmentMean_3/Rank:output:0%UnsortedSegmentMean_3/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_3/ones_1/packedPackUnsortedSegmentMean_3/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_3/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_3/ones_1Fill,UnsortedSegmentMean_3/ones_1/packed:output:0+UnsortedSegmentMean_3/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_3/concatConcatV2,UnsortedSegmentMean_3/strided_slice:output:0%UnsortedSegmentMean_3/ones_1:output:0*UnsortedSegmentMean_3/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_3/ReshapeReshape1UnsortedSegmentMean_3/UnsortedSegmentSum:output:0%UnsortedSegmentMean_3/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_3/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_3/MaximumMaximum&UnsortedSegmentMean_3/Reshape:output:0(UnsortedSegmentMean_3/Maximum/y:output:0*
T0*
_output_shapes
:╤
*UnsortedSegmentMean_3/UnsortedSegmentSum_1UnsortedSegmentSum/sequential_1/StatefulPartitionedCall_1:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_3/truedivRealDiv3UnsortedSegmentMean_3/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_3/Maximum:z:0*
T0*
_output_shapes
:С
EnsureShape_6EnsureShape!UnsortedSegmentMean_3/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         Ас
#update_ip/StatefulPartitionedCall_1StatefulPartitionedCallEnsureShape_6:output:0*update_ip/StatefulPartitionedCall:output:0update_ip_4845903update_ip_4845905update_ip_4845907*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_update_ip_layer_call_and_return_conditional_losses_4845902С
EnsureShape_7EnsureShape!UnsortedSegmentMean_2/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АС
+update_connection/StatefulPartitionedCall_1StatefulPartitionedCallEnsureShape_7:output:02update_connection/StatefulPartitionedCall:output:0update_connection_4845951update_connection_4845953update_connection_4845955*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_update_connection_layer_call_and_return_conditional_losses_4845950Q
GatherV2_8/axisConst*
_output_shapes
: *
dtype0*
value	B : ╕

GatherV2_8GatherV2,update_ip/StatefulPartitionedCall_1:output:0Squeeze_1:output:0GatherV2_8/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:Q
GatherV2_9/axisConst*
_output_shapes
: *
dtype0*
value	B : └

GatherV2_9GatherV24update_connection/StatefulPartitionedCall_1:output:0Squeeze_2:output:0GatherV2_9/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:M

Squeeze_13SqueezeGatherV2_9:output:0*
T0*
_output_shapes
:M

Squeeze_14SqueezeGatherV2_8:output:0*
T0*
_output_shapes
:O
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concat_5ConcatV2Squeeze_14:output:0Squeeze_13:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes
:Б
EnsureShape_8EnsureShapeconcat_5:output:0*
T0*(
_output_shapes
:         А*
shape:         АО
$sequential/StatefulPartitionedCall_2StatefulPartitionedCallEnsureShape_8:output:0sequential_4845802sequential_4845804*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4845441t
UnsortedSegmentMean_4/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_4/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_4/onesFill$UnsortedSegmentMean_4/Shape:output:0)UnsortedSegmentMean_4/ones/Const:output:0*
T0*
_output_shapes
:├
(UnsortedSegmentMean_4/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_4/ones:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
#UnsortedSegmentMean_4/strided_sliceStridedSlicen_c2UnsortedSegmentMean_4/strided_slice/stack:output:04UnsortedSegmentMean_4/strided_slice/stack_1:output:04UnsortedSegmentMean_4/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_4/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_4/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_4/subSub#UnsortedSegmentMean_4/Rank:output:0%UnsortedSegmentMean_4/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_4/ones_1/packedPackUnsortedSegmentMean_4/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_4/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_4/ones_1Fill,UnsortedSegmentMean_4/ones_1/packed:output:0+UnsortedSegmentMean_4/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_4/concatConcatV2,UnsortedSegmentMean_4/strided_slice:output:0%UnsortedSegmentMean_4/ones_1:output:0*UnsortedSegmentMean_4/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_4/ReshapeReshape1UnsortedSegmentMean_4/UnsortedSegmentSum:output:0%UnsortedSegmentMean_4/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_4/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_4/MaximumMaximum&UnsortedSegmentMean_4/Reshape:output:0(UnsortedSegmentMean_4/Maximum/y:output:0*
T0*
_output_shapes
:╧
*UnsortedSegmentMean_4/UnsortedSegmentSum_1UnsortedSegmentSum-sequential/StatefulPartitionedCall_2:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_4/truedivRealDiv3UnsortedSegmentMean_4/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_4/Maximum:z:0*
T0*
_output_shapes
:R
GatherV2_10/axisConst*
_output_shapes
: *
dtype0*
value	B : ┬
GatherV2_10GatherV24update_connection/StatefulPartitionedCall_1:output:0Squeeze_3:output:0GatherV2_10/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_11/axisConst*
_output_shapes
: *
dtype0*
value	B : ║
GatherV2_11GatherV2,update_ip/StatefulPartitionedCall_1:output:0Squeeze_4:output:0GatherV2_11/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_15SqueezeGatherV2_11:output:0*
T0*
_output_shapes
:N

Squeeze_16SqueezeGatherV2_10:output:0*
T0*
_output_shapes
:O
concat_6/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concat_6ConcatV2Squeeze_16:output:0Squeeze_15:output:0concat_6/axis:output:0*
N*
T0*
_output_shapes
:Б
EnsureShape_9EnsureShapeconcat_6:output:0*
T0*(
_output_shapes
:         А*
shape:         АЦ
&sequential_1/StatefulPartitionedCall_2StatefulPartitionedCallEnsureShape_9:output:0sequential_1_4845837sequential_1_4845839*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_4845526t
UnsortedSegmentMean_5/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_5/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_5/onesFill$UnsortedSegmentMean_5/Shape:output:0)UnsortedSegmentMean_5/ones/Const:output:0*
T0*
_output_shapes
:├
(UnsortedSegmentMean_5/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_5/ones:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
#UnsortedSegmentMean_5/strided_sliceStridedSlicen_i2UnsortedSegmentMean_5/strided_slice/stack:output:04UnsortedSegmentMean_5/strided_slice/stack_1:output:04UnsortedSegmentMean_5/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_5/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_5/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_5/subSub#UnsortedSegmentMean_5/Rank:output:0%UnsortedSegmentMean_5/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_5/ones_1/packedPackUnsortedSegmentMean_5/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_5/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_5/ones_1Fill,UnsortedSegmentMean_5/ones_1/packed:output:0+UnsortedSegmentMean_5/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_5/concatConcatV2,UnsortedSegmentMean_5/strided_slice:output:0%UnsortedSegmentMean_5/ones_1:output:0*UnsortedSegmentMean_5/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_5/ReshapeReshape1UnsortedSegmentMean_5/UnsortedSegmentSum:output:0%UnsortedSegmentMean_5/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_5/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_5/MaximumMaximum&UnsortedSegmentMean_5/Reshape:output:0(UnsortedSegmentMean_5/Maximum/y:output:0*
T0*
_output_shapes
:╤
*UnsortedSegmentMean_5/UnsortedSegmentSum_1UnsortedSegmentSum/sequential_1/StatefulPartitionedCall_2:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_5/truedivRealDiv3UnsortedSegmentMean_5/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_5/Maximum:z:0*
T0*
_output_shapes
:Т
EnsureShape_10EnsureShape!UnsortedSegmentMean_5/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         Аф
#update_ip/StatefulPartitionedCall_2StatefulPartitionedCallEnsureShape_10:output:0,update_ip/StatefulPartitionedCall_1:output:0update_ip_4845903update_ip_4845905update_ip_4845907*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_update_ip_layer_call_and_return_conditional_losses_4845902Т
EnsureShape_11EnsureShape!UnsortedSegmentMean_4/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АФ
+update_connection/StatefulPartitionedCall_2StatefulPartitionedCallEnsureShape_11:output:04update_connection/StatefulPartitionedCall_1:output:0update_connection_4845951update_connection_4845953update_connection_4845955*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_update_connection_layer_call_and_return_conditional_losses_4845950R
GatherV2_12/axisConst*
_output_shapes
: *
dtype0*
value	B : ║
GatherV2_12GatherV2,update_ip/StatefulPartitionedCall_2:output:0Squeeze_1:output:0GatherV2_12/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_13/axisConst*
_output_shapes
: *
dtype0*
value	B : ┬
GatherV2_13GatherV24update_connection/StatefulPartitionedCall_2:output:0Squeeze_2:output:0GatherV2_13/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_17SqueezeGatherV2_13:output:0*
T0*
_output_shapes
:N

Squeeze_18SqueezeGatherV2_12:output:0*
T0*
_output_shapes
:O
concat_7/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concat_7ConcatV2Squeeze_18:output:0Squeeze_17:output:0concat_7/axis:output:0*
N*
T0*
_output_shapes
:В
EnsureShape_12EnsureShapeconcat_7:output:0*
T0*(
_output_shapes
:         А*
shape:         АП
$sequential/StatefulPartitionedCall_3StatefulPartitionedCallEnsureShape_12:output:0sequential_4845802sequential_4845804*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4845441t
UnsortedSegmentMean_6/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_6/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_6/onesFill$UnsortedSegmentMean_6/Shape:output:0)UnsortedSegmentMean_6/ones/Const:output:0*
T0*
_output_shapes
:├
(UnsortedSegmentMean_6/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_6/ones:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
#UnsortedSegmentMean_6/strided_sliceStridedSlicen_c2UnsortedSegmentMean_6/strided_slice/stack:output:04UnsortedSegmentMean_6/strided_slice/stack_1:output:04UnsortedSegmentMean_6/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_6/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_6/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_6/subSub#UnsortedSegmentMean_6/Rank:output:0%UnsortedSegmentMean_6/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_6/ones_1/packedPackUnsortedSegmentMean_6/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_6/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_6/ones_1Fill,UnsortedSegmentMean_6/ones_1/packed:output:0+UnsortedSegmentMean_6/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_6/concatConcatV2,UnsortedSegmentMean_6/strided_slice:output:0%UnsortedSegmentMean_6/ones_1:output:0*UnsortedSegmentMean_6/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_6/ReshapeReshape1UnsortedSegmentMean_6/UnsortedSegmentSum:output:0%UnsortedSegmentMean_6/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_6/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_6/MaximumMaximum&UnsortedSegmentMean_6/Reshape:output:0(UnsortedSegmentMean_6/Maximum/y:output:0*
T0*
_output_shapes
:╧
*UnsortedSegmentMean_6/UnsortedSegmentSum_1UnsortedSegmentSum-sequential/StatefulPartitionedCall_3:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_6/truedivRealDiv3UnsortedSegmentMean_6/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_6/Maximum:z:0*
T0*
_output_shapes
:R
GatherV2_14/axisConst*
_output_shapes
: *
dtype0*
value	B : ┬
GatherV2_14GatherV24update_connection/StatefulPartitionedCall_2:output:0Squeeze_3:output:0GatherV2_14/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_15/axisConst*
_output_shapes
: *
dtype0*
value	B : ║
GatherV2_15GatherV2,update_ip/StatefulPartitionedCall_2:output:0Squeeze_4:output:0GatherV2_15/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_19SqueezeGatherV2_15:output:0*
T0*
_output_shapes
:N

Squeeze_20SqueezeGatherV2_14:output:0*
T0*
_output_shapes
:O
concat_8/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concat_8ConcatV2Squeeze_20:output:0Squeeze_19:output:0concat_8/axis:output:0*
N*
T0*
_output_shapes
:В
EnsureShape_13EnsureShapeconcat_8:output:0*
T0*(
_output_shapes
:         А*
shape:         АЧ
&sequential_1/StatefulPartitionedCall_3StatefulPartitionedCallEnsureShape_13:output:0sequential_1_4845837sequential_1_4845839*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_4845526t
UnsortedSegmentMean_7/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_7/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_7/onesFill$UnsortedSegmentMean_7/Shape:output:0)UnsortedSegmentMean_7/ones/Const:output:0*
T0*
_output_shapes
:├
(UnsortedSegmentMean_7/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_7/ones:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
#UnsortedSegmentMean_7/strided_sliceStridedSlicen_i2UnsortedSegmentMean_7/strided_slice/stack:output:04UnsortedSegmentMean_7/strided_slice/stack_1:output:04UnsortedSegmentMean_7/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_7/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_7/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_7/subSub#UnsortedSegmentMean_7/Rank:output:0%UnsortedSegmentMean_7/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_7/ones_1/packedPackUnsortedSegmentMean_7/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_7/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_7/ones_1Fill,UnsortedSegmentMean_7/ones_1/packed:output:0+UnsortedSegmentMean_7/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_7/concatConcatV2,UnsortedSegmentMean_7/strided_slice:output:0%UnsortedSegmentMean_7/ones_1:output:0*UnsortedSegmentMean_7/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_7/ReshapeReshape1UnsortedSegmentMean_7/UnsortedSegmentSum:output:0%UnsortedSegmentMean_7/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_7/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_7/MaximumMaximum&UnsortedSegmentMean_7/Reshape:output:0(UnsortedSegmentMean_7/Maximum/y:output:0*
T0*
_output_shapes
:╤
*UnsortedSegmentMean_7/UnsortedSegmentSum_1UnsortedSegmentSum/sequential_1/StatefulPartitionedCall_3:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_7/truedivRealDiv3UnsortedSegmentMean_7/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_7/Maximum:z:0*
T0*
_output_shapes
:Т
EnsureShape_14EnsureShape!UnsortedSegmentMean_7/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         Аф
#update_ip/StatefulPartitionedCall_3StatefulPartitionedCallEnsureShape_14:output:0,update_ip/StatefulPartitionedCall_2:output:0update_ip_4845903update_ip_4845905update_ip_4845907*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_update_ip_layer_call_and_return_conditional_losses_4845902Т
EnsureShape_15EnsureShape!UnsortedSegmentMean_6/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АФ
+update_connection/StatefulPartitionedCall_3StatefulPartitionedCallEnsureShape_15:output:04update_connection/StatefulPartitionedCall_2:output:0update_connection_4845951update_connection_4845953update_connection_4845955*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_update_connection_layer_call_and_return_conditional_losses_4845950R
GatherV2_16/axisConst*
_output_shapes
: *
dtype0*
value	B : ║
GatherV2_16GatherV2,update_ip/StatefulPartitionedCall_3:output:0Squeeze_1:output:0GatherV2_16/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_17/axisConst*
_output_shapes
: *
dtype0*
value	B : ┬
GatherV2_17GatherV24update_connection/StatefulPartitionedCall_3:output:0Squeeze_2:output:0GatherV2_17/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_21SqueezeGatherV2_17:output:0*
T0*
_output_shapes
:N

Squeeze_22SqueezeGatherV2_16:output:0*
T0*
_output_shapes
:O
concat_9/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concat_9ConcatV2Squeeze_22:output:0Squeeze_21:output:0concat_9/axis:output:0*
N*
T0*
_output_shapes
:В
EnsureShape_16EnsureShapeconcat_9:output:0*
T0*(
_output_shapes
:         А*
shape:         АП
$sequential/StatefulPartitionedCall_4StatefulPartitionedCallEnsureShape_16:output:0sequential_4845802sequential_4845804*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4845441t
UnsortedSegmentMean_8/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_8/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_8/onesFill$UnsortedSegmentMean_8/Shape:output:0)UnsortedSegmentMean_8/ones/Const:output:0*
T0*
_output_shapes
:├
(UnsortedSegmentMean_8/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_8/ones:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
#UnsortedSegmentMean_8/strided_sliceStridedSlicen_c2UnsortedSegmentMean_8/strided_slice/stack:output:04UnsortedSegmentMean_8/strided_slice/stack_1:output:04UnsortedSegmentMean_8/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_8/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_8/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_8/subSub#UnsortedSegmentMean_8/Rank:output:0%UnsortedSegmentMean_8/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_8/ones_1/packedPackUnsortedSegmentMean_8/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_8/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_8/ones_1Fill,UnsortedSegmentMean_8/ones_1/packed:output:0+UnsortedSegmentMean_8/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_8/concatConcatV2,UnsortedSegmentMean_8/strided_slice:output:0%UnsortedSegmentMean_8/ones_1:output:0*UnsortedSegmentMean_8/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_8/ReshapeReshape1UnsortedSegmentMean_8/UnsortedSegmentSum:output:0%UnsortedSegmentMean_8/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_8/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_8/MaximumMaximum&UnsortedSegmentMean_8/Reshape:output:0(UnsortedSegmentMean_8/Maximum/y:output:0*
T0*
_output_shapes
:╧
*UnsortedSegmentMean_8/UnsortedSegmentSum_1UnsortedSegmentSum-sequential/StatefulPartitionedCall_4:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_8/truedivRealDiv3UnsortedSegmentMean_8/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_8/Maximum:z:0*
T0*
_output_shapes
:R
GatherV2_18/axisConst*
_output_shapes
: *
dtype0*
value	B : ┬
GatherV2_18GatherV24update_connection/StatefulPartitionedCall_3:output:0Squeeze_3:output:0GatherV2_18/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_19/axisConst*
_output_shapes
: *
dtype0*
value	B : ║
GatherV2_19GatherV2,update_ip/StatefulPartitionedCall_3:output:0Squeeze_4:output:0GatherV2_19/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_23SqueezeGatherV2_19:output:0*
T0*
_output_shapes
:N

Squeeze_24SqueezeGatherV2_18:output:0*
T0*
_output_shapes
:P
concat_10/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_10ConcatV2Squeeze_24:output:0Squeeze_23:output:0concat_10/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_17EnsureShapeconcat_10:output:0*
T0*(
_output_shapes
:         А*
shape:         АЧ
&sequential_1/StatefulPartitionedCall_4StatefulPartitionedCallEnsureShape_17:output:0sequential_1_4845837sequential_1_4845839*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_4845526t
UnsortedSegmentMean_9/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧e
 UnsortedSegmentMean_9/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ц
UnsortedSegmentMean_9/onesFill$UnsortedSegmentMean_9/Shape:output:0)UnsortedSegmentMean_9/ones/Const:output:0*
T0*
_output_shapes
:├
(UnsortedSegmentMean_9/UnsortedSegmentSumUnsortedSegmentSum#UnsortedSegmentMean_9/ones:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:s
)UnsortedSegmentMean_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+UnsortedSegmentMean_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Я
#UnsortedSegmentMean_9/strided_sliceStridedSlicen_i2UnsortedSegmentMean_9/strided_slice/stack:output:04UnsortedSegmentMean_9/strided_slice/stack_1:output:04UnsortedSegmentMean_9/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask\
UnsortedSegmentMean_9/RankConst*
_output_shapes
: *
dtype0*
value	B :Y
UnsortedSegmentMean_9/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Н
UnsortedSegmentMean_9/subSub#UnsortedSegmentMean_9/Rank:output:0%UnsortedSegmentMean_9/Rank_1:output:0*
T0*
_output_shapes
: x
#UnsortedSegmentMean_9/ones_1/packedPackUnsortedSegmentMean_9/sub:z:0*
N*
T0*
_output_shapes
:d
"UnsortedSegmentMean_9/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 Rн
UnsortedSegmentMean_9/ones_1Fill,UnsortedSegmentMean_9/ones_1/packed:output:0+UnsortedSegmentMean_9/ones_1/Const:output:0*
T0	*#
_output_shapes
:         c
!UnsortedSegmentMean_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
UnsortedSegmentMean_9/concatConcatV2,UnsortedSegmentMean_9/strided_slice:output:0%UnsortedSegmentMean_9/ones_1:output:0*UnsortedSegmentMean_9/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         │
UnsortedSegmentMean_9/ReshapeReshape1UnsortedSegmentMean_9/UnsortedSegmentSum:output:0%UnsortedSegmentMean_9/concat:output:0*
Tshape0	*
T0*
_output_shapes
:d
UnsortedSegmentMean_9/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Э
UnsortedSegmentMean_9/MaximumMaximum&UnsortedSegmentMean_9/Reshape:output:0(UnsortedSegmentMean_9/Maximum/y:output:0*
T0*
_output_shapes
:╤
*UnsortedSegmentMean_9/UnsortedSegmentSum_1UnsortedSegmentSum/sequential_1/StatefulPartitionedCall_4:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:г
UnsortedSegmentMean_9/truedivRealDiv3UnsortedSegmentMean_9/UnsortedSegmentSum_1:output:0!UnsortedSegmentMean_9/Maximum:z:0*
T0*
_output_shapes
:Т
EnsureShape_18EnsureShape!UnsortedSegmentMean_9/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         Аф
#update_ip/StatefulPartitionedCall_4StatefulPartitionedCallEnsureShape_18:output:0,update_ip/StatefulPartitionedCall_3:output:0update_ip_4845903update_ip_4845905update_ip_4845907*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_update_ip_layer_call_and_return_conditional_losses_4845902Т
EnsureShape_19EnsureShape!UnsortedSegmentMean_8/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АФ
+update_connection/StatefulPartitionedCall_4StatefulPartitionedCallEnsureShape_19:output:04update_connection/StatefulPartitionedCall_3:output:0update_connection_4845951update_connection_4845953update_connection_4845955*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_update_connection_layer_call_and_return_conditional_losses_4845950R
GatherV2_20/axisConst*
_output_shapes
: *
dtype0*
value	B : ║
GatherV2_20GatherV2,update_ip/StatefulPartitionedCall_4:output:0Squeeze_1:output:0GatherV2_20/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_21/axisConst*
_output_shapes
: *
dtype0*
value	B : ┬
GatherV2_21GatherV24update_connection/StatefulPartitionedCall_4:output:0Squeeze_2:output:0GatherV2_21/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_25SqueezeGatherV2_21:output:0*
T0*
_output_shapes
:N

Squeeze_26SqueezeGatherV2_20:output:0*
T0*
_output_shapes
:P
concat_11/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_11ConcatV2Squeeze_26:output:0Squeeze_25:output:0concat_11/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_20EnsureShapeconcat_11:output:0*
T0*(
_output_shapes
:         А*
shape:         АП
$sequential/StatefulPartitionedCall_5StatefulPartitionedCallEnsureShape_20:output:0sequential_4845802sequential_4845804*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4845441u
UnsortedSegmentMean_10/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧f
!UnsortedSegmentMean_10/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
UnsortedSegmentMean_10/onesFill%UnsortedSegmentMean_10/Shape:output:0*UnsortedSegmentMean_10/ones/Const:output:0*
T0*
_output_shapes
:┼
)UnsortedSegmentMean_10/UnsortedSegmentSumUnsortedSegmentSum$UnsortedSegmentMean_10/ones:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:t
*UnsortedSegmentMean_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
$UnsortedSegmentMean_10/strided_sliceStridedSlicen_c3UnsortedSegmentMean_10/strided_slice/stack:output:05UnsortedSegmentMean_10/strided_slice/stack_1:output:05UnsortedSegmentMean_10/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask]
UnsortedSegmentMean_10/RankConst*
_output_shapes
: *
dtype0*
value	B :Z
UnsortedSegmentMean_10/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Р
UnsortedSegmentMean_10/subSub$UnsortedSegmentMean_10/Rank:output:0&UnsortedSegmentMean_10/Rank_1:output:0*
T0*
_output_shapes
: z
$UnsortedSegmentMean_10/ones_1/packedPackUnsortedSegmentMean_10/sub:z:0*
N*
T0*
_output_shapes
:e
#UnsortedSegmentMean_10/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R░
UnsortedSegmentMean_10/ones_1Fill-UnsortedSegmentMean_10/ones_1/packed:output:0,UnsortedSegmentMean_10/ones_1/Const:output:0*
T0	*#
_output_shapes
:         d
"UnsortedSegmentMean_10/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
UnsortedSegmentMean_10/concatConcatV2-UnsortedSegmentMean_10/strided_slice:output:0&UnsortedSegmentMean_10/ones_1:output:0+UnsortedSegmentMean_10/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         ╢
UnsortedSegmentMean_10/ReshapeReshape2UnsortedSegmentMean_10/UnsortedSegmentSum:output:0&UnsortedSegmentMean_10/concat:output:0*
Tshape0	*
T0*
_output_shapes
:e
 UnsortedSegmentMean_10/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
UnsortedSegmentMean_10/MaximumMaximum'UnsortedSegmentMean_10/Reshape:output:0)UnsortedSegmentMean_10/Maximum/y:output:0*
T0*
_output_shapes
:╨
+UnsortedSegmentMean_10/UnsortedSegmentSum_1UnsortedSegmentSum-sequential/StatefulPartitionedCall_5:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:ж
UnsortedSegmentMean_10/truedivRealDiv4UnsortedSegmentMean_10/UnsortedSegmentSum_1:output:0"UnsortedSegmentMean_10/Maximum:z:0*
T0*
_output_shapes
:R
GatherV2_22/axisConst*
_output_shapes
: *
dtype0*
value	B : ┬
GatherV2_22GatherV24update_connection/StatefulPartitionedCall_4:output:0Squeeze_3:output:0GatherV2_22/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_23/axisConst*
_output_shapes
: *
dtype0*
value	B : ║
GatherV2_23GatherV2,update_ip/StatefulPartitionedCall_4:output:0Squeeze_4:output:0GatherV2_23/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_27SqueezeGatherV2_23:output:0*
T0*
_output_shapes
:N

Squeeze_28SqueezeGatherV2_22:output:0*
T0*
_output_shapes
:P
concat_12/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_12ConcatV2Squeeze_28:output:0Squeeze_27:output:0concat_12/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_21EnsureShapeconcat_12:output:0*
T0*(
_output_shapes
:         А*
shape:         АЧ
&sequential_1/StatefulPartitionedCall_5StatefulPartitionedCallEnsureShape_21:output:0sequential_1_4845837sequential_1_4845839*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_4845526u
UnsortedSegmentMean_11/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧f
!UnsortedSegmentMean_11/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
UnsortedSegmentMean_11/onesFill%UnsortedSegmentMean_11/Shape:output:0*UnsortedSegmentMean_11/ones/Const:output:0*
T0*
_output_shapes
:┼
)UnsortedSegmentMean_11/UnsortedSegmentSumUnsortedSegmentSum$UnsortedSegmentMean_11/ones:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:t
*UnsortedSegmentMean_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
$UnsortedSegmentMean_11/strided_sliceStridedSlicen_i3UnsortedSegmentMean_11/strided_slice/stack:output:05UnsortedSegmentMean_11/strided_slice/stack_1:output:05UnsortedSegmentMean_11/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask]
UnsortedSegmentMean_11/RankConst*
_output_shapes
: *
dtype0*
value	B :Z
UnsortedSegmentMean_11/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Р
UnsortedSegmentMean_11/subSub$UnsortedSegmentMean_11/Rank:output:0&UnsortedSegmentMean_11/Rank_1:output:0*
T0*
_output_shapes
: z
$UnsortedSegmentMean_11/ones_1/packedPackUnsortedSegmentMean_11/sub:z:0*
N*
T0*
_output_shapes
:e
#UnsortedSegmentMean_11/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R░
UnsortedSegmentMean_11/ones_1Fill-UnsortedSegmentMean_11/ones_1/packed:output:0,UnsortedSegmentMean_11/ones_1/Const:output:0*
T0	*#
_output_shapes
:         d
"UnsortedSegmentMean_11/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
UnsortedSegmentMean_11/concatConcatV2-UnsortedSegmentMean_11/strided_slice:output:0&UnsortedSegmentMean_11/ones_1:output:0+UnsortedSegmentMean_11/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         ╢
UnsortedSegmentMean_11/ReshapeReshape2UnsortedSegmentMean_11/UnsortedSegmentSum:output:0&UnsortedSegmentMean_11/concat:output:0*
Tshape0	*
T0*
_output_shapes
:e
 UnsortedSegmentMean_11/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
UnsortedSegmentMean_11/MaximumMaximum'UnsortedSegmentMean_11/Reshape:output:0)UnsortedSegmentMean_11/Maximum/y:output:0*
T0*
_output_shapes
:╥
+UnsortedSegmentMean_11/UnsortedSegmentSum_1UnsortedSegmentSum/sequential_1/StatefulPartitionedCall_5:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:ж
UnsortedSegmentMean_11/truedivRealDiv4UnsortedSegmentMean_11/UnsortedSegmentSum_1:output:0"UnsortedSegmentMean_11/Maximum:z:0*
T0*
_output_shapes
:У
EnsureShape_22EnsureShape"UnsortedSegmentMean_11/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         Аф
#update_ip/StatefulPartitionedCall_5StatefulPartitionedCallEnsureShape_22:output:0,update_ip/StatefulPartitionedCall_4:output:0update_ip_4845903update_ip_4845905update_ip_4845907*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_update_ip_layer_call_and_return_conditional_losses_4845902У
EnsureShape_23EnsureShape"UnsortedSegmentMean_10/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АФ
+update_connection/StatefulPartitionedCall_5StatefulPartitionedCallEnsureShape_23:output:04update_connection/StatefulPartitionedCall_4:output:0update_connection_4845951update_connection_4845953update_connection_4845955*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_update_connection_layer_call_and_return_conditional_losses_4845950R
GatherV2_24/axisConst*
_output_shapes
: *
dtype0*
value	B : ║
GatherV2_24GatherV2,update_ip/StatefulPartitionedCall_5:output:0Squeeze_1:output:0GatherV2_24/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_25/axisConst*
_output_shapes
: *
dtype0*
value	B : ┬
GatherV2_25GatherV24update_connection/StatefulPartitionedCall_5:output:0Squeeze_2:output:0GatherV2_25/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_29SqueezeGatherV2_25:output:0*
T0*
_output_shapes
:N

Squeeze_30SqueezeGatherV2_24:output:0*
T0*
_output_shapes
:P
concat_13/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_13ConcatV2Squeeze_30:output:0Squeeze_29:output:0concat_13/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_24EnsureShapeconcat_13:output:0*
T0*(
_output_shapes
:         А*
shape:         АП
$sequential/StatefulPartitionedCall_6StatefulPartitionedCallEnsureShape_24:output:0sequential_4845802sequential_4845804*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4845441u
UnsortedSegmentMean_12/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧f
!UnsortedSegmentMean_12/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
UnsortedSegmentMean_12/onesFill%UnsortedSegmentMean_12/Shape:output:0*UnsortedSegmentMean_12/ones/Const:output:0*
T0*
_output_shapes
:┼
)UnsortedSegmentMean_12/UnsortedSegmentSumUnsortedSegmentSum$UnsortedSegmentMean_12/ones:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:t
*UnsortedSegmentMean_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
$UnsortedSegmentMean_12/strided_sliceStridedSlicen_c3UnsortedSegmentMean_12/strided_slice/stack:output:05UnsortedSegmentMean_12/strided_slice/stack_1:output:05UnsortedSegmentMean_12/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask]
UnsortedSegmentMean_12/RankConst*
_output_shapes
: *
dtype0*
value	B :Z
UnsortedSegmentMean_12/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Р
UnsortedSegmentMean_12/subSub$UnsortedSegmentMean_12/Rank:output:0&UnsortedSegmentMean_12/Rank_1:output:0*
T0*
_output_shapes
: z
$UnsortedSegmentMean_12/ones_1/packedPackUnsortedSegmentMean_12/sub:z:0*
N*
T0*
_output_shapes
:e
#UnsortedSegmentMean_12/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R░
UnsortedSegmentMean_12/ones_1Fill-UnsortedSegmentMean_12/ones_1/packed:output:0,UnsortedSegmentMean_12/ones_1/Const:output:0*
T0	*#
_output_shapes
:         d
"UnsortedSegmentMean_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
UnsortedSegmentMean_12/concatConcatV2-UnsortedSegmentMean_12/strided_slice:output:0&UnsortedSegmentMean_12/ones_1:output:0+UnsortedSegmentMean_12/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         ╢
UnsortedSegmentMean_12/ReshapeReshape2UnsortedSegmentMean_12/UnsortedSegmentSum:output:0&UnsortedSegmentMean_12/concat:output:0*
Tshape0	*
T0*
_output_shapes
:e
 UnsortedSegmentMean_12/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
UnsortedSegmentMean_12/MaximumMaximum'UnsortedSegmentMean_12/Reshape:output:0)UnsortedSegmentMean_12/Maximum/y:output:0*
T0*
_output_shapes
:╨
+UnsortedSegmentMean_12/UnsortedSegmentSum_1UnsortedSegmentSum-sequential/StatefulPartitionedCall_6:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:ж
UnsortedSegmentMean_12/truedivRealDiv4UnsortedSegmentMean_12/UnsortedSegmentSum_1:output:0"UnsortedSegmentMean_12/Maximum:z:0*
T0*
_output_shapes
:R
GatherV2_26/axisConst*
_output_shapes
: *
dtype0*
value	B : ┬
GatherV2_26GatherV24update_connection/StatefulPartitionedCall_5:output:0Squeeze_3:output:0GatherV2_26/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_27/axisConst*
_output_shapes
: *
dtype0*
value	B : ║
GatherV2_27GatherV2,update_ip/StatefulPartitionedCall_5:output:0Squeeze_4:output:0GatherV2_27/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_31SqueezeGatherV2_27:output:0*
T0*
_output_shapes
:N

Squeeze_32SqueezeGatherV2_26:output:0*
T0*
_output_shapes
:P
concat_14/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_14ConcatV2Squeeze_32:output:0Squeeze_31:output:0concat_14/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_25EnsureShapeconcat_14:output:0*
T0*(
_output_shapes
:         А*
shape:         АЧ
&sequential_1/StatefulPartitionedCall_6StatefulPartitionedCallEnsureShape_25:output:0sequential_1_4845837sequential_1_4845839*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_4845526u
UnsortedSegmentMean_13/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧f
!UnsortedSegmentMean_13/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
UnsortedSegmentMean_13/onesFill%UnsortedSegmentMean_13/Shape:output:0*UnsortedSegmentMean_13/ones/Const:output:0*
T0*
_output_shapes
:┼
)UnsortedSegmentMean_13/UnsortedSegmentSumUnsortedSegmentSum$UnsortedSegmentMean_13/ones:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:t
*UnsortedSegmentMean_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
$UnsortedSegmentMean_13/strided_sliceStridedSlicen_i3UnsortedSegmentMean_13/strided_slice/stack:output:05UnsortedSegmentMean_13/strided_slice/stack_1:output:05UnsortedSegmentMean_13/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask]
UnsortedSegmentMean_13/RankConst*
_output_shapes
: *
dtype0*
value	B :Z
UnsortedSegmentMean_13/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Р
UnsortedSegmentMean_13/subSub$UnsortedSegmentMean_13/Rank:output:0&UnsortedSegmentMean_13/Rank_1:output:0*
T0*
_output_shapes
: z
$UnsortedSegmentMean_13/ones_1/packedPackUnsortedSegmentMean_13/sub:z:0*
N*
T0*
_output_shapes
:e
#UnsortedSegmentMean_13/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R░
UnsortedSegmentMean_13/ones_1Fill-UnsortedSegmentMean_13/ones_1/packed:output:0,UnsortedSegmentMean_13/ones_1/Const:output:0*
T0	*#
_output_shapes
:         d
"UnsortedSegmentMean_13/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
UnsortedSegmentMean_13/concatConcatV2-UnsortedSegmentMean_13/strided_slice:output:0&UnsortedSegmentMean_13/ones_1:output:0+UnsortedSegmentMean_13/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         ╢
UnsortedSegmentMean_13/ReshapeReshape2UnsortedSegmentMean_13/UnsortedSegmentSum:output:0&UnsortedSegmentMean_13/concat:output:0*
Tshape0	*
T0*
_output_shapes
:e
 UnsortedSegmentMean_13/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
UnsortedSegmentMean_13/MaximumMaximum'UnsortedSegmentMean_13/Reshape:output:0)UnsortedSegmentMean_13/Maximum/y:output:0*
T0*
_output_shapes
:╥
+UnsortedSegmentMean_13/UnsortedSegmentSum_1UnsortedSegmentSum/sequential_1/StatefulPartitionedCall_6:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:ж
UnsortedSegmentMean_13/truedivRealDiv4UnsortedSegmentMean_13/UnsortedSegmentSum_1:output:0"UnsortedSegmentMean_13/Maximum:z:0*
T0*
_output_shapes
:У
EnsureShape_26EnsureShape"UnsortedSegmentMean_13/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         Аф
#update_ip/StatefulPartitionedCall_6StatefulPartitionedCallEnsureShape_26:output:0,update_ip/StatefulPartitionedCall_5:output:0update_ip_4845903update_ip_4845905update_ip_4845907*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_update_ip_layer_call_and_return_conditional_losses_4845902У
EnsureShape_27EnsureShape"UnsortedSegmentMean_12/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АФ
+update_connection/StatefulPartitionedCall_6StatefulPartitionedCallEnsureShape_27:output:04update_connection/StatefulPartitionedCall_5:output:0update_connection_4845951update_connection_4845953update_connection_4845955*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_update_connection_layer_call_and_return_conditional_losses_4845950R
GatherV2_28/axisConst*
_output_shapes
: *
dtype0*
value	B : ║
GatherV2_28GatherV2,update_ip/StatefulPartitionedCall_6:output:0Squeeze_1:output:0GatherV2_28/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_29/axisConst*
_output_shapes
: *
dtype0*
value	B : ┬
GatherV2_29GatherV24update_connection/StatefulPartitionedCall_6:output:0Squeeze_2:output:0GatherV2_29/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_33SqueezeGatherV2_29:output:0*
T0*
_output_shapes
:N

Squeeze_34SqueezeGatherV2_28:output:0*
T0*
_output_shapes
:P
concat_15/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_15ConcatV2Squeeze_34:output:0Squeeze_33:output:0concat_15/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_28EnsureShapeconcat_15:output:0*
T0*(
_output_shapes
:         А*
shape:         АП
$sequential/StatefulPartitionedCall_7StatefulPartitionedCallEnsureShape_28:output:0sequential_4845802sequential_4845804*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4845441u
UnsortedSegmentMean_14/ShapeShapeSqueeze_2:output:0*
T0	*#
_output_shapes
:         :э╧f
!UnsortedSegmentMean_14/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
UnsortedSegmentMean_14/onesFill%UnsortedSegmentMean_14/Shape:output:0*UnsortedSegmentMean_14/ones/Const:output:0*
T0*
_output_shapes
:┼
)UnsortedSegmentMean_14/UnsortedSegmentSumUnsortedSegmentSum$UnsortedSegmentMean_14/ones:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:t
*UnsortedSegmentMean_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
$UnsortedSegmentMean_14/strided_sliceStridedSlicen_c3UnsortedSegmentMean_14/strided_slice/stack:output:05UnsortedSegmentMean_14/strided_slice/stack_1:output:05UnsortedSegmentMean_14/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask]
UnsortedSegmentMean_14/RankConst*
_output_shapes
: *
dtype0*
value	B :Z
UnsortedSegmentMean_14/Rank_1RankSqueeze_2:output:0*
T0	*
_output_shapes
: Р
UnsortedSegmentMean_14/subSub$UnsortedSegmentMean_14/Rank:output:0&UnsortedSegmentMean_14/Rank_1:output:0*
T0*
_output_shapes
: z
$UnsortedSegmentMean_14/ones_1/packedPackUnsortedSegmentMean_14/sub:z:0*
N*
T0*
_output_shapes
:e
#UnsortedSegmentMean_14/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R░
UnsortedSegmentMean_14/ones_1Fill-UnsortedSegmentMean_14/ones_1/packed:output:0,UnsortedSegmentMean_14/ones_1/Const:output:0*
T0	*#
_output_shapes
:         d
"UnsortedSegmentMean_14/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
UnsortedSegmentMean_14/concatConcatV2-UnsortedSegmentMean_14/strided_slice:output:0&UnsortedSegmentMean_14/ones_1:output:0+UnsortedSegmentMean_14/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         ╢
UnsortedSegmentMean_14/ReshapeReshape2UnsortedSegmentMean_14/UnsortedSegmentSum:output:0&UnsortedSegmentMean_14/concat:output:0*
Tshape0	*
T0*
_output_shapes
:e
 UnsortedSegmentMean_14/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
UnsortedSegmentMean_14/MaximumMaximum'UnsortedSegmentMean_14/Reshape:output:0)UnsortedSegmentMean_14/Maximum/y:output:0*
T0*
_output_shapes
:╨
+UnsortedSegmentMean_14/UnsortedSegmentSum_1UnsortedSegmentSum-sequential/StatefulPartitionedCall_7:output:0Squeeze_2:output:0n_c*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:ж
UnsortedSegmentMean_14/truedivRealDiv4UnsortedSegmentMean_14/UnsortedSegmentSum_1:output:0"UnsortedSegmentMean_14/Maximum:z:0*
T0*
_output_shapes
:R
GatherV2_30/axisConst*
_output_shapes
: *
dtype0*
value	B : ┬
GatherV2_30GatherV24update_connection/StatefulPartitionedCall_6:output:0Squeeze_3:output:0GatherV2_30/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:R
GatherV2_31/axisConst*
_output_shapes
: *
dtype0*
value	B : ║
GatherV2_31GatherV2,update_ip/StatefulPartitionedCall_6:output:0Squeeze_4:output:0GatherV2_31/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*
_output_shapes
:N

Squeeze_35SqueezeGatherV2_31:output:0*
T0*
_output_shapes
:N

Squeeze_36SqueezeGatherV2_30:output:0*
T0*
_output_shapes
:P
concat_16/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
	concat_16ConcatV2Squeeze_36:output:0Squeeze_35:output:0concat_16/axis:output:0*
N*
T0*
_output_shapes
:Г
EnsureShape_29EnsureShapeconcat_16:output:0*
T0*(
_output_shapes
:         А*
shape:         АЧ
&sequential_1/StatefulPartitionedCall_7StatefulPartitionedCallEnsureShape_29:output:0sequential_1_4845837sequential_1_4845839*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_4845526u
UnsortedSegmentMean_15/ShapeShapeSqueeze_4:output:0*
T0	*#
_output_shapes
:         :э╧f
!UnsortedSegmentMean_15/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Щ
UnsortedSegmentMean_15/onesFill%UnsortedSegmentMean_15/Shape:output:0*UnsortedSegmentMean_15/ones/Const:output:0*
T0*
_output_shapes
:┼
)UnsortedSegmentMean_15/UnsortedSegmentSumUnsortedSegmentSum$UnsortedSegmentMean_15/ones:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:t
*UnsortedSegmentMean_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,UnsortedSegmentMean_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
$UnsortedSegmentMean_15/strided_sliceStridedSlicen_i3UnsortedSegmentMean_15/strided_slice/stack:output:05UnsortedSegmentMean_15/strided_slice/stack_1:output:05UnsortedSegmentMean_15/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
new_axis_mask]
UnsortedSegmentMean_15/RankConst*
_output_shapes
: *
dtype0*
value	B :Z
UnsortedSegmentMean_15/Rank_1RankSqueeze_4:output:0*
T0	*
_output_shapes
: Р
UnsortedSegmentMean_15/subSub$UnsortedSegmentMean_15/Rank:output:0&UnsortedSegmentMean_15/Rank_1:output:0*
T0*
_output_shapes
: z
$UnsortedSegmentMean_15/ones_1/packedPackUnsortedSegmentMean_15/sub:z:0*
N*
T0*
_output_shapes
:e
#UnsortedSegmentMean_15/ones_1/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R░
UnsortedSegmentMean_15/ones_1Fill-UnsortedSegmentMean_15/ones_1/packed:output:0,UnsortedSegmentMean_15/ones_1/Const:output:0*
T0	*#
_output_shapes
:         d
"UnsortedSegmentMean_15/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
UnsortedSegmentMean_15/concatConcatV2-UnsortedSegmentMean_15/strided_slice:output:0&UnsortedSegmentMean_15/ones_1:output:0+UnsortedSegmentMean_15/concat/axis:output:0*
N*
T0	*#
_output_shapes
:         ╢
UnsortedSegmentMean_15/ReshapeReshape2UnsortedSegmentMean_15/UnsortedSegmentSum:output:0&UnsortedSegmentMean_15/concat:output:0*
Tshape0	*
T0*
_output_shapes
:e
 UnsortedSegmentMean_15/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?а
UnsortedSegmentMean_15/MaximumMaximum'UnsortedSegmentMean_15/Reshape:output:0)UnsortedSegmentMean_15/Maximum/y:output:0*
T0*
_output_shapes
:╥
+UnsortedSegmentMean_15/UnsortedSegmentSum_1UnsortedSegmentSum/sequential_1/StatefulPartitionedCall_7:output:0Squeeze_4:output:0n_i*
Tindices0	*
Tnumsegments0	*
T0*
_output_shapes
:ж
UnsortedSegmentMean_15/truedivRealDiv4UnsortedSegmentMean_15/UnsortedSegmentSum_1:output:0"UnsortedSegmentMean_15/Maximum:z:0*
T0*
_output_shapes
:У
EnsureShape_30EnsureShape"UnsortedSegmentMean_15/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         Аф
#update_ip/StatefulPartitionedCall_7StatefulPartitionedCallEnsureShape_30:output:0,update_ip/StatefulPartitionedCall_6:output:0update_ip_4845903update_ip_4845905update_ip_4845907*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_update_ip_layer_call_and_return_conditional_losses_4845902У
EnsureShape_31EnsureShape"UnsortedSegmentMean_14/truediv:z:0*
T0*(
_output_shapes
:         А*
shape:         АФ
+update_connection/StatefulPartitionedCall_7StatefulPartitionedCallEnsureShape_31:output:04update_connection/StatefulPartitionedCall_6:output:0update_connection_4845951update_connection_4845953update_connection_4845955*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_update_connection_layer_call_and_return_conditional_losses_4845950С
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall4update_connection/StatefulPartitionedCall_7:output:0sequential_2_4846505sequential_2_4846507sequential_2_4846509sequential_2_4846511sequential_2_4846513sequential_2_4846515*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_4845656|
IdentityIdentity-sequential_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         с

NoOpNoOp#^sequential/StatefulPartitionedCall%^sequential/StatefulPartitionedCall_1%^sequential/StatefulPartitionedCall_2%^sequential/StatefulPartitionedCall_3%^sequential/StatefulPartitionedCall_4%^sequential/StatefulPartitionedCall_5%^sequential/StatefulPartitionedCall_6%^sequential/StatefulPartitionedCall_7%^sequential_1/StatefulPartitionedCall'^sequential_1/StatefulPartitionedCall_1'^sequential_1/StatefulPartitionedCall_2'^sequential_1/StatefulPartitionedCall_3'^sequential_1/StatefulPartitionedCall_4'^sequential_1/StatefulPartitionedCall_5'^sequential_1/StatefulPartitionedCall_6'^sequential_1/StatefulPartitionedCall_7%^sequential_2/StatefulPartitionedCall*^update_connection/StatefulPartitionedCall,^update_connection/StatefulPartitionedCall_1,^update_connection/StatefulPartitionedCall_2,^update_connection/StatefulPartitionedCall_3,^update_connection/StatefulPartitionedCall_4,^update_connection/StatefulPartitionedCall_5,^update_connection/StatefulPartitionedCall_6,^update_connection/StatefulPartitionedCall_7"^update_ip/StatefulPartitionedCall$^update_ip/StatefulPartitionedCall_1$^update_ip/StatefulPartitionedCall_2$^update_ip/StatefulPartitionedCall_3$^update_ip/StatefulPartitionedCall_4$^update_ip/StatefulPartitionedCall_5$^update_ip/StatefulPartitionedCall_6$^update_ip/StatefulPartitionedCall_7*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:::: : ::: : : : : : : : : : : : : : : : 2L
$sequential/StatefulPartitionedCall_1$sequential/StatefulPartitionedCall_12L
$sequential/StatefulPartitionedCall_2$sequential/StatefulPartitionedCall_22L
$sequential/StatefulPartitionedCall_3$sequential/StatefulPartitionedCall_32L
$sequential/StatefulPartitionedCall_4$sequential/StatefulPartitionedCall_42L
$sequential/StatefulPartitionedCall_5$sequential/StatefulPartitionedCall_52L
$sequential/StatefulPartitionedCall_6$sequential/StatefulPartitionedCall_62L
$sequential/StatefulPartitionedCall_7$sequential/StatefulPartitionedCall_72H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2P
&sequential_1/StatefulPartitionedCall_1&sequential_1/StatefulPartitionedCall_12P
&sequential_1/StatefulPartitionedCall_2&sequential_1/StatefulPartitionedCall_22P
&sequential_1/StatefulPartitionedCall_3&sequential_1/StatefulPartitionedCall_32P
&sequential_1/StatefulPartitionedCall_4&sequential_1/StatefulPartitionedCall_42P
&sequential_1/StatefulPartitionedCall_5&sequential_1/StatefulPartitionedCall_52P
&sequential_1/StatefulPartitionedCall_6&sequential_1/StatefulPartitionedCall_62P
&sequential_1/StatefulPartitionedCall_7&sequential_1/StatefulPartitionedCall_72L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2Z
+update_connection/StatefulPartitionedCall_1+update_connection/StatefulPartitionedCall_12Z
+update_connection/StatefulPartitionedCall_2+update_connection/StatefulPartitionedCall_22Z
+update_connection/StatefulPartitionedCall_3+update_connection/StatefulPartitionedCall_32Z
+update_connection/StatefulPartitionedCall_4+update_connection/StatefulPartitionedCall_42Z
+update_connection/StatefulPartitionedCall_5+update_connection/StatefulPartitionedCall_52Z
+update_connection/StatefulPartitionedCall_6+update_connection/StatefulPartitionedCall_62Z
+update_connection/StatefulPartitionedCall_7+update_connection/StatefulPartitionedCall_72V
)update_connection/StatefulPartitionedCall)update_connection/StatefulPartitionedCall2J
#update_ip/StatefulPartitionedCall_1#update_ip/StatefulPartitionedCall_12J
#update_ip/StatefulPartitionedCall_2#update_ip/StatefulPartitionedCall_22J
#update_ip/StatefulPartitionedCall_3#update_ip/StatefulPartitionedCall_32J
#update_ip/StatefulPartitionedCall_4#update_ip/StatefulPartitionedCall_42J
#update_ip/StatefulPartitionedCall_5#update_ip/StatefulPartitionedCall_52J
#update_ip/StatefulPartitionedCall_6#update_ip/StatefulPartitionedCall_62J
#update_ip/StatefulPartitionedCall_7#update_ip/StatefulPartitionedCall_72F
!update_ip/StatefulPartitionedCall!update_ip/StatefulPartitionedCall:'#
!
_user_specified_name	4846515:'#
!
_user_specified_name	4846513:'#
!
_user_specified_name	4846511:'#
!
_user_specified_name	4846509:'#
!
_user_specified_name	4846507:'#
!
_user_specified_name	4846505:'#
!
_user_specified_name	4845955:'#
!
_user_specified_name	4845953:'#
!
_user_specified_name	4845951:'#
!
_user_specified_name	4845907:'#
!
_user_specified_name	4845905:'#
!
_user_specified_name	4845903:'
#
!
_user_specified_name	4845839:'	#
!
_user_specified_name	4845837:'#
!
_user_specified_name	4845804:'#
!
_user_specified_name	4845802:NJ

_output_shapes
:
.
_user_specified_namesrc_ip_to_connection:NJ

_output_shapes
:
.
_user_specified_namesrc_connection_to_ip:;7

_output_shapes
: 

_user_specified_namen_i:;7

_output_shapes
: 

_user_specified_namen_c:LH

_output_shapes
:
,
_user_specified_namefeature_connection:NJ

_output_shapes
:
.
_user_specified_namedst_ip_to_connection:N J

_output_shapes
:
.
_user_specified_namedst_connection_to_ip
╥
╖
%__inference_gnn_layer_call_fn_4847313
dst_connection_to_ip	
dst_ip_to_connection	
feature_connection
n_c	
n_i	
src_connection_to_ip	
src_ip_to_connection	
unknown:
АА
	unknown_0:	А
	unknown_1:
АА
	unknown_2:	А
	unknown_3:	А
	unknown_4:
АА
	unknown_5:
АА
	unknown_6:	А
	unknown_7:
АА
	unknown_8:
АА
	unknown_9:
АА

unknown_10:	А

unknown_11:	А@

unknown_12:@

unknown_13:@

unknown_14:
identityИвStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCalldst_connection_to_ipdst_ip_to_connectionfeature_connectionn_cn_isrc_connection_to_ipsrc_ip_to_connectionunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*"
Tin
2						*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_gnn_layer_call_and_return_conditional_losses_4846519o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:::: : ::: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	4847309:'#
!
_user_specified_name	4847307:'#
!
_user_specified_name	4847305:'#
!
_user_specified_name	4847303:'#
!
_user_specified_name	4847301:'#
!
_user_specified_name	4847299:'#
!
_user_specified_name	4847297:'#
!
_user_specified_name	4847295:'#
!
_user_specified_name	4847293:'#
!
_user_specified_name	4847291:'#
!
_user_specified_name	4847289:'#
!
_user_specified_name	4847287:'
#
!
_user_specified_name	4847285:'	#
!
_user_specified_name	4847283:'#
!
_user_specified_name	4847281:'#
!
_user_specified_name	4847279:NJ

_output_shapes
:
.
_user_specified_namesrc_ip_to_connection:NJ

_output_shapes
:
.
_user_specified_namesrc_connection_to_ip:;7

_output_shapes
: 

_user_specified_namen_i:;7

_output_shapes
: 

_user_specified_namen_c:LH

_output_shapes
:
,
_user_specified_namefeature_connection:NJ

_output_shapes
:
.
_user_specified_namedst_ip_to_connection:N J

_output_shapes
:
.
_user_specified_namedst_connection_to_ip
А
Э
,__inference_sequential_layer_call_fn_4845465
input_1
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCall▐
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4845441p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	4845461:'#
!
_user_specified_name	4845459:Q M
(
_output_shapes
:         А
!
_user_specified_name	input_1
є
┌
F__inference_update_ip_layer_call_and_return_conditional_losses_4846653

inputs

states*
readvariableop_resource:	А2
matmul_readvariableop_resource:
АА4
 matmul_1_readvariableop_resource:
АА
identity

identity_1ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	А*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:А:А*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         АZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         АZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"А   А       \
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╔
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         АN
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         Аc
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         АR
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         А^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         АZ
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         АJ
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:         АT
mul_1MulSigmoid:y:0states*
T0*(
_output_shapes
:         АJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         АR
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:         АW
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         АY
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         А[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         Аe
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         А:         А: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:PL
(
_output_shapes
:         А
 
_user_specified_namestates:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
б

e
F__inference_dropout_2_layer_call_and_return_conditional_losses_4847804

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
│
ф
3__inference_update_connection_layer_call_fn_4847576

inputs
states_0
unknown:	А
	unknown_0:
АА
	unknown_1:
АА
identity

identity_1ИвStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_update_connection_layer_call_and_return_conditional_losses_4845950p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         Аr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         А:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	4847570:'#
!
_user_specified_name	4847568:'#
!
_user_specified_name	4847566:ZV
0
_output_shapes
:                  
"
_user_specified_name
states_0:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
е
G
+__inference_dropout_1_layer_call_fn_4847725

inputs
identity▓
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_4845533a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
▌
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_4847809

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
├
ъ
G__inference_sequential_layer_call_and_return_conditional_losses_4845441
input_1!
dense_4845435:
АА
dense_4845437:	А
identityИвdense/StatefulPartitionedCallвdropout/StatefulPartitionedCall╔
dropout/StatefulPartitionedCallStatefulPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_4845422К
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_4845435dense_4845437*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_4845434v
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         Аd
NoOpNoOp^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:'#
!
_user_specified_name	4845437:'#
!
_user_specified_name	4845435:Q M
(
_output_shapes
:         А
!
_user_specified_name	input_1
╒

Ў
B__inference_dense_layer_call_and_return_conditional_losses_4845434

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         АS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┘
d
F__inference_dropout_3_layer_call_and_return_conditional_losses_4845679

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         @[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╫

°
D__inference_dense_1_layer_call_and_return_conditional_losses_4847762

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         АS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
√
▄
F__inference_update_ip_layer_call_and_return_conditional_losses_4847523

inputs
states_0*
readvariableop_resource:	А2
matmul_readvariableop_resource:
АА4
 matmul_1_readvariableop_resource:
АА
identity

identity_1ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	А*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:А:А*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         АZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         АZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"А   А       \
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╔
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         АN
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         Аc
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         АR
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         А^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         АZ
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         АJ
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:         АV
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:         АJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         АR
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:         АW
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         АY
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         А[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         Аe
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         А:         А: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:RN
(
_output_shapes
:         А
"
_user_specified_name
states_0:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
▌
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_4845533

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
█
b
D__inference_dropout_layer_call_and_return_conditional_losses_4847695

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╧

М
.__inference_sequential_2_layer_call_fn_4845704
input_3
unknown:
АА
	unknown_0:	А
	unknown_1:	А@
	unknown_2:@
	unknown_3:@
	unknown_4:
identityИвStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_4845656o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	4845700:'#
!
_user_specified_name	4845698:'#
!
_user_specified_name	4845696:'#
!
_user_specified_name	4845694:'#
!
_user_specified_name	4845692:'#
!
_user_specified_name	4845690:Q M
(
_output_shapes
:         А
!
_user_specified_name	input_3
є
Ч
)__inference_dense_3_layer_call_fn_4847818

inputs
unknown:	А@
	unknown_0:@
identityИвStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_4845620o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	4847814:'#
!
_user_specified_name	4847812:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╧

М
.__inference_sequential_2_layer_call_fn_4845721
input_3
unknown:
АА
	unknown_0:	А
	unknown_1:	А@
	unknown_2:@
	unknown_3:@
	unknown_4:
identityИвStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_4845687o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	4845717:'#
!
_user_specified_name	4845715:'#
!
_user_specified_name	4845713:'#
!
_user_specified_name	4845711:'#
!
_user_specified_name	4845709:'#
!
_user_specified_name	4845707:Q M
(
_output_shapes
:         А
!
_user_specified_name	input_3
Д
Я
.__inference_sequential_1_layer_call_fn_4845559
input_2
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_4845541p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	4845555:'#
!
_user_specified_name	4845553:Q M
(
_output_shapes
:         А
!
_user_specified_name	input_2
Я

c
D__inference_dropout_layer_call_and_return_conditional_losses_4845422

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
█
b
D__inference_dropout_layer_call_and_return_conditional_losses_4845448

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
е
G
+__inference_dropout_2_layer_call_fn_4847792

inputs
identity▓
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_4845668a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╙
d
+__inference_dropout_1_layer_call_fn_4847720

inputs
identityИвStatefulPartitionedCall┬
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_4845507p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
У
▄
+__inference_update_ip_layer_call_fn_4847484

inputs
states_0
unknown:	А
	unknown_0:
АА
	unknown_1:
АА
identity

identity_1ИвStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:         А:         А*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_update_ip_layer_call_and_return_conditional_losses_4846653p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         Аr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         А:         А: : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	4847478:'#
!
_user_specified_name	4847476:'#
!
_user_specified_name	4847474:RN
(
_output_shapes
:         А
"
_user_specified_name
states_0:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╥
╖
%__inference_gnn_layer_call_fn_4847356
dst_connection_to_ip	
dst_ip_to_connection	
feature_connection
n_c	
n_i	
src_connection_to_ip	
src_ip_to_connection	
unknown:
АА
	unknown_0:	А
	unknown_1:
АА
	unknown_2:	А
	unknown_3:	А
	unknown_4:
АА
	unknown_5:
АА
	unknown_6:	А
	unknown_7:
АА
	unknown_8:
АА
	unknown_9:
АА

unknown_10:	А

unknown_11:	А@

unknown_12:@

unknown_13:@

unknown_14:
identityИвStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCalldst_connection_to_ipdst_ip_to_connectionfeature_connectionn_cn_isrc_connection_to_ipsrc_ip_to_connectionunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*"
Tin
2						*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_gnn_layer_call_and_return_conditional_losses_4847270o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:::: : ::: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	4847352:'#
!
_user_specified_name	4847350:'#
!
_user_specified_name	4847348:'#
!
_user_specified_name	4847346:'#
!
_user_specified_name	4847344:'#
!
_user_specified_name	4847342:'#
!
_user_specified_name	4847340:'#
!
_user_specified_name	4847338:'#
!
_user_specified_name	4847336:'#
!
_user_specified_name	4847334:'#
!
_user_specified_name	4847332:'#
!
_user_specified_name	4847330:'
#
!
_user_specified_name	4847328:'	#
!
_user_specified_name	4847326:'#
!
_user_specified_name	4847324:'#
!
_user_specified_name	4847322:NJ

_output_shapes
:
.
_user_specified_namesrc_ip_to_connection:NJ

_output_shapes
:
.
_user_specified_namesrc_connection_to_ip:;7

_output_shapes
: 

_user_specified_namen_i:;7

_output_shapes
: 

_user_specified_namen_c:LH

_output_shapes
:
,
_user_specified_namefeature_connection:NJ

_output_shapes
:
.
_user_specified_namedst_ip_to_connection:N J

_output_shapes
:
.
_user_specified_namedst_connection_to_ip
Я

c
D__inference_dropout_layer_call_and_return_conditional_losses_4847690

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АQ
dropout/ShapeShapeinputs*
T0*
_output_shapes
::э╧Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
А
Э
,__inference_sequential_layer_call_fn_4845474
input_1
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCall▐
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_4845456p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	4845470:'#
!
_user_specified_name	4845468:Q M
(
_output_shapes
:         А
!
_user_specified_name	input_1
°
с
I__inference_sequential_2_layer_call_and_return_conditional_losses_4845656
input_3#
dense_2_4845592:
АА
dense_2_4845594:	А"
dense_3_4845621:	А@
dense_3_4845623:@!
dense_4_4845650:@
dense_4_4845652:
identityИвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallв!dropout_2/StatefulPartitionedCallв!dropout_3/StatefulPartitionedCallё
dense_2/StatefulPartitionedCallStatefulPartitionedCallinput_3dense_2_4845592dense_2_4845594*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_4845591ю
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_4845608У
dense_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_3_4845621dense_3_4845623*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_4845620С
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_dropout_3_layer_call_and_return_conditional_losses_4845637У
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_4_4845650dense_4_4845652*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_4845649w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ╨
NoOpNoOp ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         А: : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall:'#
!
_user_specified_name	4845652:'#
!
_user_specified_name	4845650:'#
!
_user_specified_name	4845623:'#
!
_user_specified_name	4845621:'#
!
_user_specified_name	4845594:'#
!
_user_specified_name	4845592:Q M
(
_output_shapes
:         А
!
_user_specified_name	input_3
У
ф
N__inference_update_connection_layer_call_and_return_conditional_losses_4847629

inputs
states_0*
readvariableop_resource:	А2
matmul_readvariableop_resource:
АА4
 matmul_1_readvariableop_resource:
АА
identity

identity_1ИвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpвReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	А*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:А:А*	
numv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:         АZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ж
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splitz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:         АZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"А   А       \
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
         ╔
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*

Tlen0*
T0*P
_output_shapes>
<:         А:         А:         А*
	num_splita
addAddV2split:output:0split_1:output:0*
T0*(
_output_shapes
:         АN
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:         Аc
add_1AddV2split:output:1split_1:output:1*
T0*(
_output_shapes
:         АR
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:         А^
mulMulSigmoid_1:y:0split_1:output:2*
T0*(
_output_shapes
:         АZ
add_2AddV2split:output:2mul:z:0*
T0*(
_output_shapes
:         АJ
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:         АV
mul_1MulSigmoid:y:0states_0*
T0*(
_output_shapes
:         АJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Z
subSubsub/x:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         АR
mul_2Mulsub:z:0Tanh:y:0*
T0*(
_output_shapes
:         АW
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*(
_output_shapes
:         АY
IdentityIdentity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         А[

Identity_1Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:         Аe
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:         А:                  : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:ZV
0
_output_shapes
:                  
"
_user_specified_name
states_0:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs"эL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ъ
serving_defaultЖ
F
dst_connection_to_ip.
&serving_default_dst_connection_to_ip:0	
F
dst_ip_to_connection.
&serving_default_dst_ip_to_connection:0	
B
feature_connection,
$serving_default_feature_connection:0
"
n_c
serving_default_n_c:0	 
"
n_i
serving_default_n_i:0	 
F
src_connection_to_ip.
&serving_default_src_connection_to_ip:0	
F
src_ip_to_connection.
&serving_default_src_ip_to_connection:0	<
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict:┼ы
─
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	ip_update
	connection_update

message_func1
message_func2
readout
	optimizer
call

signatures"
_tf_keras_model
Ц
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15"
trackable_list_wrapper
Ц
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
 non_trainable_variables

!layers
"metrics
#layer_regularization_losses
$layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
╜
%trace_0
&trace_12Ж
%__inference_gnn_layer_call_fn_4847313
%__inference_gnn_layer_call_fn_4847356╡
о▓к
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 z%trace_0z&trace_1
є
'trace_0
(trace_12╝
@__inference_gnn_layer_call_and_return_conditional_losses_4846519
@__inference_gnn_layer_call_and_return_conditional_losses_4847270╡
о▓к
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 z'trace_0z(trace_1
║B╖
"__inference__wrapped_model_4845408dst_connection_to_ipdst_ip_to_connectionfeature_connectionn_cn_isrc_connection_to_ipsrc_ip_to_connection"Ш
С▓Н
FullArgSpec
argsЪ

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ш
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses
/_random_generator

kernel
recurrent_kernel
bias"
_tf_keras_layer
ш
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6_random_generator

kernel
recurrent_kernel
bias"
_tf_keras_layer
▐
7layer-0
8layer_with_weights-0
8layer-1
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_sequential
▐
?layer-0
@layer_with_weights-0
@layer-1
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses"
_tf_keras_sequential
╣
Glayer_with_weights-0
Glayer-0
Hlayer-1
Ilayer_with_weights-1
Ilayer-2
Jlayer-3
Klayer_with_weights-2
Klayer-4
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
P__call__
*Q&call_and_return_all_conditional_losses"
_tf_keras_sequential
д
R
_variables
S_iterations
T_current_learning_rate
U_index_dict
V
_momentums
W_velocities
X_update_step_xla"
experimentalOptimizer
╤
Ytrace_02┤
__inference_call_560424Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zYtrace_0
,
Zserving_default"
signature_map
$:"
АА2update_ip/kernel
.:,
АА2update_ip/recurrent_kernel
!:	А2update_ip/bias
,:*
АА2update_connection/kernel
6:4
АА2"update_connection/recurrent_kernel
):'	А2update_connection/bias
 :
АА2dense/kernel
:А2
dense/bias
": 
АА2dense_1/kernel
:А2dense_1/bias
": 
АА2dense_2/kernel
:А2dense_2/bias
!:	А@2dense_3/kernel
:@2dense_3/bias
 :@2dense_4/kernel
:2dense_4/bias
 "
trackable_list_wrapper
C
0
	1

2
3
4"
trackable_list_wrapper
о
[0
\1
]2
^3
_4
`5
a6
b7
c8
d9
e10
f11
g12
h13
i14
j15
k16
l17
m18
n19
o20
p21
q22
r23
s24
t25
u26
v27
w28
x29
y30
z31
{32
|33
}34"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╔B╞
%__inference_gnn_layer_call_fn_4847313dst_connection_to_ipdst_ip_to_connectionfeature_connectionn_cn_isrc_connection_to_ipsrc_ip_to_connection"д
Э▓Щ
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining
kwonlydefaults
 
annotationsк *
 
╔B╞
%__inference_gnn_layer_call_fn_4847356dst_connection_to_ipdst_ip_to_connectionfeature_connectionn_cn_isrc_connection_to_ipsrc_ip_to_connection"д
Э▓Щ
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining
kwonlydefaults
 
annotationsк *
 
фBс
@__inference_gnn_layer_call_and_return_conditional_losses_4846519dst_connection_to_ipdst_ip_to_connectionfeature_connectionn_cn_isrc_connection_to_ipsrc_ip_to_connection"д
Э▓Щ
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining
kwonlydefaults
 
annotationsк *
 
фBс
@__inference_gnn_layer_call_and_return_conditional_losses_4847270dst_connection_to_ipdst_ip_to_connectionfeature_connectionn_cn_isrc_connection_to_ipsrc_ip_to_connection"д
Э▓Щ
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ

jtraining
kwonlydefaults
 
annotationsк *
 
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
░
~non_trainable_variables

layers
Аmetrics
 Бlayer_regularization_losses
Вlayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
╦
Гtrace_0
Дtrace_12Р
+__inference_update_ip_layer_call_fn_4847470
+__inference_update_ip_layer_call_fn_4847484│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zГtrace_0zДtrace_1
Б
Еtrace_0
Жtrace_12╞
F__inference_update_ip_layer_call_and_return_conditional_losses_4847523
F__inference_update_ip_layer_call_and_return_conditional_losses_4847562│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЕtrace_0zЖtrace_1
"
_generic_user_object
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
█
Мtrace_0
Нtrace_12а
3__inference_update_connection_layer_call_fn_4847576
3__inference_update_connection_layer_call_fn_4847590│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zМtrace_0zНtrace_1
С
Оtrace_0
Пtrace_12╓
N__inference_update_connection_layer_call_and_return_conditional_losses_4847629
N__inference_update_connection_layer_call_and_return_conditional_losses_4847668│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zОtrace_0zПtrace_1
"
_generic_user_object
├
Р	variables
Сtrainable_variables
Тregularization_losses
У	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses
Ц_random_generator"
_tf_keras_layer
┴
Ч	variables
Шtrainable_variables
Щregularization_losses
Ъ	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
╧
вtrace_0
гtrace_12Ф
,__inference_sequential_layer_call_fn_4845465
,__inference_sequential_layer_call_fn_4845474╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zвtrace_0zгtrace_1
Е
дtrace_0
еtrace_12╩
G__inference_sequential_layer_call_and_return_conditional_losses_4845441
G__inference_sequential_layer_call_and_return_conditional_losses_4845456╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zдtrace_0zеtrace_1
├
ж	variables
зtrainable_variables
иregularization_losses
й	keras_api
к__call__
+л&call_and_return_all_conditional_losses
м_random_generator"
_tf_keras_layer
┴
н	variables
оtrainable_variables
пregularization_losses
░	keras_api
▒__call__
+▓&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
│non_trainable_variables
┤layers
╡metrics
 ╢layer_regularization_losses
╖layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
╙
╕trace_0
╣trace_12Ш
.__inference_sequential_1_layer_call_fn_4845550
.__inference_sequential_1_layer_call_fn_4845559╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╕trace_0z╣trace_1
Й
║trace_0
╗trace_12╬
I__inference_sequential_1_layer_call_and_return_conditional_losses_4845526
I__inference_sequential_1_layer_call_and_return_conditional_losses_4845541╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z║trace_0z╗trace_1
┴
╝	variables
╜trainable_variables
╛regularization_losses
┐	keras_api
└__call__
+┴&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
├
┬	variables
├trainable_variables
─regularization_losses
┼	keras_api
╞__call__
+╟&call_and_return_all_conditional_losses
╚_random_generator"
_tf_keras_layer
┴
╔	variables
╩trainable_variables
╦regularization_losses
╠	keras_api
═__call__
+╬&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
├
╧	variables
╨trainable_variables
╤regularization_losses
╥	keras_api
╙__call__
+╘&call_and_return_all_conditional_losses
╒_random_generator"
_tf_keras_layer
┴
╓	variables
╫trainable_variables
╪regularization_losses
┘	keras_api
┌__call__
+█&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
▄non_trainable_variables
▌layers
▐metrics
 ▀layer_regularization_losses
рlayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
P__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
╙
сtrace_0
тtrace_12Ш
.__inference_sequential_2_layer_call_fn_4845704
.__inference_sequential_2_layer_call_fn_4845721╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zсtrace_0zтtrace_1
Й
уtrace_0
фtrace_12╬
I__inference_sequential_2_layer_call_and_return_conditional_losses_4845656
I__inference_sequential_2_layer_call_and_return_conditional_losses_4845687╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zуtrace_0zфtrace_1
╛
S0
х1
ц2
ч3
ш4
щ5
ъ6
ы7
ь8
э9
ю10
я11
Ё12
ё13
Є14
є15
Ї16
ї17
Ў18
ў19
°20
∙21
·22
√23
№24
¤25
■26
 27
А28
Б29
В30
Г31
Д32"
trackable_list_wrapper
:	 2	iteration
: 2current_learning_rate
 "
trackable_dict_wrapper
ж
х0
ч1
щ2
ы3
э4
я5
ё6
є7
ї8
ў9
∙10
√11
¤12
 13
Б14
Г15"
trackable_list_wrapper
ж
ц0
ш1
ъ2
ь3
ю4
Ё5
Є6
Ї7
Ў8
°9
·10
№11
■12
А13
В14
Д15"
trackable_list_wrapper
╡2▓п
ж▓в
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
рB▌
__inference_call_560424inputs_dst_connection_to_ipinputs_dst_ip_to_connectioninputs_feature_connection
inputs_n_c
inputs_n_iinputs_src_connection_to_ipinputs_src_ip_to_connection"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╕B╡
%__inference_signature_wrapper_4847456dst_connection_to_ipdst_ip_to_connectionfeature_connectionn_cn_isrc_connection_to_ipsrc_ip_to_connection"Х
О▓К
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 Ч

kwonlyargsИЪД
jdst_connection_to_ip
jdst_ip_to_connection
jfeature_connection
jn_c
jn_i
jsrc_connection_to_ip
jsrc_ip_to_connection
kwonlydefaults
 
annotationsк *
 
R
Е	variables
Ж	keras_api

Зtotal

Иcount"
_tf_keras_metric
c
Й	variables
К	keras_api

Лtotal

Мcount
Н
_fn_kwargs"
_tf_keras_metric
б
О	variables
П	keras_api
Рtrue_positives
Сtrue_negatives
Тfalse_positives
Уfalse_negatives
Ф
thresholds"
_tf_keras_metric
v
Х	variables
Ц	keras_api
Ч
thresholds
Шtrue_positives
Щfalse_negatives"
_tf_keras_metric
v
Ъ	variables
Ы	keras_api
Ь
thresholds
Эtrue_positives
Юfalse_positives"
_tf_keras_metric
v
Я	variables
а	keras_api
б
thresholds
вtrue_positives
гfalse_negatives"
_tf_keras_metric
v
д	variables
е	keras_api
ж
thresholds
зtrue_positives
иfalse_positives"
_tf_keras_metric
v
й	variables
к	keras_api
л
thresholds
мtrue_positives
нfalse_negatives"
_tf_keras_metric
v
о	variables
п	keras_api
░
thresholds
▒true_positives
▓false_positives"
_tf_keras_metric
v
│	variables
┤	keras_api
╡
thresholds
╢true_positives
╖false_negatives"
_tf_keras_metric
v
╕	variables
╣	keras_api
║
thresholds
╗true_positives
╝false_positives"
_tf_keras_metric
v
╜	variables
╛	keras_api
┐
thresholds
└true_positives
┴false_negatives"
_tf_keras_metric
v
┬	variables
├	keras_api
─
thresholds
┼true_positives
╞false_positives"
_tf_keras_metric
v
╟	variables
╚	keras_api
╔
thresholds
╩true_positives
╦false_negatives"
_tf_keras_metric
v
╠	variables
═	keras_api
╬
thresholds
╧true_positives
╨false_positives"
_tf_keras_metric
v
╤	variables
╥	keras_api
╙
thresholds
╘true_positives
╒false_negatives"
_tf_keras_metric
v
╓	variables
╫	keras_api
╪
thresholds
┘true_positives
┌false_positives"
_tf_keras_metric
v
█	variables
▄	keras_api
▌
thresholds
▐true_positives
▀false_negatives"
_tf_keras_metric
v
р	variables
с	keras_api
т
thresholds
уtrue_positives
фfalse_positives"
_tf_keras_metric
v
х	variables
ц	keras_api
ч
thresholds
шtrue_positives
щfalse_negatives"
_tf_keras_metric
v
ъ	variables
ы	keras_api
ь
thresholds
эtrue_positives
юfalse_positives"
_tf_keras_metric
v
я	variables
Ё	keras_api
ё
thresholds
Єtrue_positives
єfalse_negatives"
_tf_keras_metric
v
Ї	variables
ї	keras_api
Ў
thresholds
ўtrue_positives
°false_positives"
_tf_keras_metric
v
∙	variables
·	keras_api
√
thresholds
№true_positives
¤false_negatives"
_tf_keras_metric
v
■	variables
 	keras_api
А
thresholds
Бtrue_positives
Вfalse_positives"
_tf_keras_metric
v
Г	variables
Д	keras_api
Е
thresholds
Жtrue_positives
Зfalse_negatives"
_tf_keras_metric
v
И	variables
Й	keras_api
К
thresholds
Лtrue_positives
Мfalse_positives"
_tf_keras_metric
v
Н	variables
О	keras_api
П
thresholds
Рtrue_positives
Сfalse_negatives"
_tf_keras_metric
v
Т	variables
У	keras_api
Ф
thresholds
Хtrue_positives
Цfalse_positives"
_tf_keras_metric
v
Ч	variables
Ш	keras_api
Щ
thresholds
Ъtrue_positives
Ыfalse_negatives"
_tf_keras_metric
v
Ь	variables
Э	keras_api
Ю
thresholds
Яtrue_positives
аfalse_positives"
_tf_keras_metric
v
б	variables
в	keras_api
г
thresholds
дtrue_positives
еfalse_negatives"
_tf_keras_metric
v
ж	variables
з	keras_api
и
thresholds
йtrue_positives
кfalse_positives"
_tf_keras_metric
з
л	variables
м	keras_api
н
init_shape
оtrue_positives
пfalse_positives
░false_negatives
▒weights_intermediate"
_tf_keras_metric
з
▓	variables
│	keras_api
┤
init_shape
╡true_positives
╢false_positives
╖false_negatives
╕weights_intermediate"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
їBЄ
+__inference_update_ip_layer_call_fn_4847470inputsstates_0"о
з▓г
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
+__inference_update_ip_layer_call_fn_4847484inputsstates_0"о
з▓г
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
РBН
F__inference_update_ip_layer_call_and_return_conditional_losses_4847523inputsstates_0"о
з▓г
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
РBН
F__inference_update_ip_layer_call_and_return_conditional_losses_4847562inputsstates_0"о
з▓г
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
¤B·
3__inference_update_connection_layer_call_fn_4847576inputsstates_0"о
з▓г
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
¤B·
3__inference_update_connection_layer_call_fn_4847590inputsstates_0"о
з▓г
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ШBХ
N__inference_update_connection_layer_call_and_return_conditional_losses_4847629inputsstates_0"о
з▓г
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ШBХ
N__inference_update_connection_layer_call_and_return_conditional_losses_4847668inputsstates_0"о
з▓г
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╣non_trainable_variables
║layers
╗metrics
 ╝layer_regularization_losses
╜layer_metrics
Р	variables
Сtrainable_variables
Тregularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
╜
╛trace_0
┐trace_12В
)__inference_dropout_layer_call_fn_4847673
)__inference_dropout_layer_call_fn_4847678й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╛trace_0z┐trace_1
є
└trace_0
┴trace_12╕
D__inference_dropout_layer_call_and_return_conditional_losses_4847690
D__inference_dropout_layer_call_and_return_conditional_losses_4847695й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z└trace_0z┴trace_1
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
┬non_trainable_variables
├layers
─metrics
 ┼layer_regularization_losses
╞layer_metrics
Ч	variables
Шtrainable_variables
Щregularization_losses
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
у
╟trace_02─
'__inference_dense_layer_call_fn_4847704Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╟trace_0
■
╚trace_02▀
B__inference_dense_layer_call_and_return_conditional_losses_4847715Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╚trace_0
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ыBш
,__inference_sequential_layer_call_fn_4845465input_1"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ыBш
,__inference_sequential_layer_call_fn_4845474input_1"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЖBГ
G__inference_sequential_layer_call_and_return_conditional_losses_4845441input_1"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЖBГ
G__inference_sequential_layer_call_and_return_conditional_losses_4845456input_1"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╔non_trainable_variables
╩layers
╦metrics
 ╠layer_regularization_losses
═layer_metrics
ж	variables
зtrainable_variables
иregularization_losses
к__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
┴
╬trace_0
╧trace_12Ж
+__inference_dropout_1_layer_call_fn_4847720
+__inference_dropout_1_layer_call_fn_4847725й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╬trace_0z╧trace_1
ў
╨trace_0
╤trace_12╝
F__inference_dropout_1_layer_call_and_return_conditional_losses_4847737
F__inference_dropout_1_layer_call_and_return_conditional_losses_4847742й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╨trace_0z╤trace_1
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╥non_trainable_variables
╙layers
╘metrics
 ╒layer_regularization_losses
╓layer_metrics
н	variables
оtrainable_variables
пregularization_losses
▒__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses"
_generic_user_object
х
╫trace_02╞
)__inference_dense_1_layer_call_fn_4847751Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╫trace_0
А
╪trace_02с
D__inference_dense_1_layer_call_and_return_conditional_losses_4847762Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╪trace_0
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
эBъ
.__inference_sequential_1_layer_call_fn_4845550input_2"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
.__inference_sequential_1_layer_call_fn_4845559input_2"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ИBЕ
I__inference_sequential_1_layer_call_and_return_conditional_losses_4845526input_2"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ИBЕ
I__inference_sequential_1_layer_call_and_return_conditional_losses_4845541input_2"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
┘non_trainable_variables
┌layers
█metrics
 ▄layer_regularization_losses
▌layer_metrics
╝	variables
╜trainable_variables
╛regularization_losses
└__call__
+┴&call_and_return_all_conditional_losses
'┴"call_and_return_conditional_losses"
_generic_user_object
х
▐trace_02╞
)__inference_dense_2_layer_call_fn_4847771Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▐trace_0
А
▀trace_02с
D__inference_dense_2_layer_call_and_return_conditional_losses_4847782Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▀trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
рnon_trainable_variables
сlayers
тmetrics
 уlayer_regularization_losses
фlayer_metrics
┬	variables
├trainable_variables
─regularization_losses
╞__call__
+╟&call_and_return_all_conditional_losses
'╟"call_and_return_conditional_losses"
_generic_user_object
┴
хtrace_0
цtrace_12Ж
+__inference_dropout_2_layer_call_fn_4847787
+__inference_dropout_2_layer_call_fn_4847792й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zхtrace_0zцtrace_1
ў
чtrace_0
шtrace_12╝
F__inference_dropout_2_layer_call_and_return_conditional_losses_4847804
F__inference_dropout_2_layer_call_and_return_conditional_losses_4847809й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zчtrace_0zшtrace_1
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
щnon_trainable_variables
ъlayers
ыmetrics
 ьlayer_regularization_losses
эlayer_metrics
╔	variables
╩trainable_variables
╦regularization_losses
═__call__
+╬&call_and_return_all_conditional_losses
'╬"call_and_return_conditional_losses"
_generic_user_object
х
юtrace_02╞
)__inference_dense_3_layer_call_fn_4847818Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zюtrace_0
А
яtrace_02с
D__inference_dense_3_layer_call_and_return_conditional_losses_4847829Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zяtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Ёnon_trainable_variables
ёlayers
Єmetrics
 єlayer_regularization_losses
Їlayer_metrics
╧	variables
╨trainable_variables
╤regularization_losses
╙__call__
+╘&call_and_return_all_conditional_losses
'╘"call_and_return_conditional_losses"
_generic_user_object
┴
їtrace_0
Ўtrace_12Ж
+__inference_dropout_3_layer_call_fn_4847834
+__inference_dropout_3_layer_call_fn_4847839й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zїtrace_0zЎtrace_1
ў
ўtrace_0
°trace_12╝
F__inference_dropout_3_layer_call_and_return_conditional_losses_4847851
F__inference_dropout_3_layer_call_and_return_conditional_losses_4847856й
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zўtrace_0z°trace_1
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
∙non_trainable_variables
·layers
√metrics
 №layer_regularization_losses
¤layer_metrics
╓	variables
╫trainable_variables
╪regularization_losses
┌__call__
+█&call_and_return_all_conditional_losses
'█"call_and_return_conditional_losses"
_generic_user_object
х
■trace_02╞
)__inference_dense_4_layer_call_fn_4847865Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z■trace_0
А
 trace_02с
D__inference_dense_4_layer_call_and_return_conditional_losses_4847876Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z trace_0
 "
trackable_list_wrapper
C
G0
H1
I2
J3
K4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
эBъ
.__inference_sequential_2_layer_call_fn_4845704input_3"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
.__inference_sequential_2_layer_call_fn_4845721input_3"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ИBЕ
I__inference_sequential_2_layer_call_and_return_conditional_losses_4845656input_3"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ИBЕ
I__inference_sequential_2_layer_call_and_return_conditional_losses_4845687input_3"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
):'
АА2Adam/m/update_ip/kernel
):'
АА2Adam/v/update_ip/kernel
3:1
АА2!Adam/m/update_ip/recurrent_kernel
3:1
АА2!Adam/v/update_ip/recurrent_kernel
&:$	А2Adam/m/update_ip/bias
&:$	А2Adam/v/update_ip/bias
1:/
АА2Adam/m/update_connection/kernel
1:/
АА2Adam/v/update_connection/kernel
;:9
АА2)Adam/m/update_connection/recurrent_kernel
;:9
АА2)Adam/v/update_connection/recurrent_kernel
.:,	А2Adam/m/update_connection/bias
.:,	А2Adam/v/update_connection/bias
%:#
АА2Adam/m/dense/kernel
%:#
АА2Adam/v/dense/kernel
:А2Adam/m/dense/bias
:А2Adam/v/dense/bias
':%
АА2Adam/m/dense_1/kernel
':%
АА2Adam/v/dense_1/kernel
 :А2Adam/m/dense_1/bias
 :А2Adam/v/dense_1/bias
':%
АА2Adam/m/dense_2/kernel
':%
АА2Adam/v/dense_2/kernel
 :А2Adam/m/dense_2/bias
 :А2Adam/v/dense_2/bias
&:$	А@2Adam/m/dense_3/kernel
&:$	А@2Adam/v/dense_3/kernel
:@2Adam/m/dense_3/bias
:@2Adam/v/dense_3/bias
%:#@2Adam/m/dense_4/kernel
%:#@2Adam/v/dense_4/kernel
:2Adam/m/dense_4/bias
:2Adam/v/dense_4/bias
0
З0
И1"
trackable_list_wrapper
.
Е	variables"
_generic_user_object
:  (2total
:  (2count
0
Л0
М1"
trackable_list_wrapper
.
Й	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
@
Р0
С1
Т2
У3"
trackable_list_wrapper
.
О	variables"
_generic_user_object
:╚ (2true_positives
:╚ (2true_negatives
 :╚ (2false_positives
 :╚ (2false_negatives
 "
trackable_list_wrapper
0
Ш0
Щ1"
trackable_list_wrapper
.
Х	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
Э0
Ю1"
trackable_list_wrapper
.
Ъ	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
в0
г1"
trackable_list_wrapper
.
Я	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
з0
и1"
trackable_list_wrapper
.
д	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
м0
н1"
trackable_list_wrapper
.
й	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
▒0
▓1"
trackable_list_wrapper
.
о	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
╢0
╖1"
trackable_list_wrapper
.
│	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
╗0
╝1"
trackable_list_wrapper
.
╕	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
└0
┴1"
trackable_list_wrapper
.
╜	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
┼0
╞1"
trackable_list_wrapper
.
┬	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
╩0
╦1"
trackable_list_wrapper
.
╟	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
╧0
╨1"
trackable_list_wrapper
.
╠	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
╘0
╒1"
trackable_list_wrapper
.
╤	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
┘0
┌1"
trackable_list_wrapper
.
╓	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
▐0
▀1"
trackable_list_wrapper
.
█	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
у0
ф1"
trackable_list_wrapper
.
р	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
ш0
щ1"
trackable_list_wrapper
.
х	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
э0
ю1"
trackable_list_wrapper
.
ъ	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
Є0
є1"
trackable_list_wrapper
.
я	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
ў0
°1"
trackable_list_wrapper
.
Ї	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
№0
¤1"
trackable_list_wrapper
.
∙	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
Б0
В1"
trackable_list_wrapper
.
■	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
Ж0
З1"
trackable_list_wrapper
.
Г	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
Л0
М1"
trackable_list_wrapper
.
И	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
Р0
С1"
trackable_list_wrapper
.
Н	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
Х0
Ц1"
trackable_list_wrapper
.
Т	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
Ъ0
Ы1"
trackable_list_wrapper
.
Ч	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
Я0
а1"
trackable_list_wrapper
.
Ь	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
д0
е1"
trackable_list_wrapper
.
б	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
0
й0
к1"
trackable_list_wrapper
.
ж	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
@
о0
п1
░2
▒3"
trackable_list_wrapper
.
л	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
: (2false_negatives
$:" (2weights_intermediate
@
╡0
╢1
╖2
╕3"
trackable_list_wrapper
.
▓	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
: (2false_negatives
$:" (2weights_intermediate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▀B▄
)__inference_dropout_layer_call_fn_4847673inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▀B▄
)__inference_dropout_layer_call_fn_4847678inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
D__inference_dropout_layer_call_and_return_conditional_losses_4847690inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
D__inference_dropout_layer_call_and_return_conditional_losses_4847695inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╤B╬
'__inference_dense_layer_call_fn_4847704inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ьBщ
B__inference_dense_layer_call_and_return_conditional_losses_4847715inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сB▐
+__inference_dropout_1_layer_call_fn_4847720inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
сB▐
+__inference_dropout_1_layer_call_fn_4847725inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
F__inference_dropout_1_layer_call_and_return_conditional_losses_4847737inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
F__inference_dropout_1_layer_call_and_return_conditional_losses_4847742inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╙B╨
)__inference_dense_1_layer_call_fn_4847751inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
юBы
D__inference_dense_1_layer_call_and_return_conditional_losses_4847762inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╙B╨
)__inference_dense_2_layer_call_fn_4847771inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
юBы
D__inference_dense_2_layer_call_and_return_conditional_losses_4847782inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сB▐
+__inference_dropout_2_layer_call_fn_4847787inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
сB▐
+__inference_dropout_2_layer_call_fn_4847792inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
F__inference_dropout_2_layer_call_and_return_conditional_losses_4847804inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
F__inference_dropout_2_layer_call_and_return_conditional_losses_4847809inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╙B╨
)__inference_dense_3_layer_call_fn_4847818inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
юBы
D__inference_dense_3_layer_call_and_return_conditional_losses_4847829inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
сB▐
+__inference_dropout_3_layer_call_fn_4847834inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
сB▐
+__inference_dropout_3_layer_call_fn_4847839inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
F__inference_dropout_3_layer_call_and_return_conditional_losses_4847851inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
№B∙
F__inference_dropout_3_layer_call_and_return_conditional_losses_4847856inputs"д
Э▓Щ
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╙B╨
)__inference_dense_4_layer_call_fn_4847865inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
юBы
D__inference_dense_4_layer_call_and_return_conditional_losses_4847876inputs"Ш
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ╦
"__inference__wrapped_model_4845408д┌в╓
╬в╩
╟к├
7
dst_connection_to_ipК
dst_connection_to_ip	
7
dst_ip_to_connectionК
dst_ip_to_connection	
3
feature_connectionК
feature_connection

n_cК	
n_c 	

n_iК	
n_i 	
7
src_connection_to_ipК
src_connection_to_ip	
7
src_ip_to_connectionК
src_ip_to_connection	
к "3к0
.
output_1"К
output_1         ▀
__inference_call_560424├ЛвЗ
 в√
°кЇ
>
dst_connection_to_ip&К#
inputs_dst_connection_to_ip	
>
dst_ip_to_connection&К#
inputs_dst_ip_to_connection	
:
feature_connection$К!
inputs_feature_connection

n_cК

inputs_n_c 	

n_iК

inputs_n_i 	
>
src_connection_to_ip&К#
inputs_src_connection_to_ip	
>
src_ip_to_connection&К#
inputs_src_ip_to_connection	
к "!К
unknown         н
D__inference_dense_1_layer_call_and_return_conditional_losses_4847762e0в-
&в#
!К
inputs         А
к "-в*
#К 
tensor_0         А
Ъ З
)__inference_dense_1_layer_call_fn_4847751Z0в-
&в#
!К
inputs         А
к ""К
unknown         Ан
D__inference_dense_2_layer_call_and_return_conditional_losses_4847782e0в-
&в#
!К
inputs         А
к "-в*
#К 
tensor_0         А
Ъ З
)__inference_dense_2_layer_call_fn_4847771Z0в-
&в#
!К
inputs         А
к ""К
unknown         Ам
D__inference_dense_3_layer_call_and_return_conditional_losses_4847829d0в-
&в#
!К
inputs         А
к ",в)
"К
tensor_0         @
Ъ Ж
)__inference_dense_3_layer_call_fn_4847818Y0в-
&в#
!К
inputs         А
к "!К
unknown         @л
D__inference_dense_4_layer_call_and_return_conditional_losses_4847876c/в,
%в"
 К
inputs         @
к ",в)
"К
tensor_0         
Ъ Е
)__inference_dense_4_layer_call_fn_4847865X/в,
%в"
 К
inputs         @
к "!К
unknown         л
B__inference_dense_layer_call_and_return_conditional_losses_4847715e0в-
&в#
!К
inputs         А
к "-в*
#К 
tensor_0         А
Ъ Е
'__inference_dense_layer_call_fn_4847704Z0в-
&в#
!К
inputs         А
к ""К
unknown         Ап
F__inference_dropout_1_layer_call_and_return_conditional_losses_4847737e4в1
*в'
!К
inputs         А
p
к "-в*
#К 
tensor_0         А
Ъ п
F__inference_dropout_1_layer_call_and_return_conditional_losses_4847742e4в1
*в'
!К
inputs         А
p 
к "-в*
#К 
tensor_0         А
Ъ Й
+__inference_dropout_1_layer_call_fn_4847720Z4в1
*в'
!К
inputs         А
p
к ""К
unknown         АЙ
+__inference_dropout_1_layer_call_fn_4847725Z4в1
*в'
!К
inputs         А
p 
к ""К
unknown         Ап
F__inference_dropout_2_layer_call_and_return_conditional_losses_4847804e4в1
*в'
!К
inputs         А
p
к "-в*
#К 
tensor_0         А
Ъ п
F__inference_dropout_2_layer_call_and_return_conditional_losses_4847809e4в1
*в'
!К
inputs         А
p 
к "-в*
#К 
tensor_0         А
Ъ Й
+__inference_dropout_2_layer_call_fn_4847787Z4в1
*в'
!К
inputs         А
p
к ""К
unknown         АЙ
+__inference_dropout_2_layer_call_fn_4847792Z4в1
*в'
!К
inputs         А
p 
к ""К
unknown         Ан
F__inference_dropout_3_layer_call_and_return_conditional_losses_4847851c3в0
)в&
 К
inputs         @
p
к ",в)
"К
tensor_0         @
Ъ н
F__inference_dropout_3_layer_call_and_return_conditional_losses_4847856c3в0
)в&
 К
inputs         @
p 
к ",в)
"К
tensor_0         @
Ъ З
+__inference_dropout_3_layer_call_fn_4847834X3в0
)в&
 К
inputs         @
p
к "!К
unknown         @З
+__inference_dropout_3_layer_call_fn_4847839X3в0
)в&
 К
inputs         @
p 
к "!К
unknown         @н
D__inference_dropout_layer_call_and_return_conditional_losses_4847690e4в1
*в'
!К
inputs         А
p
к "-в*
#К 
tensor_0         А
Ъ н
D__inference_dropout_layer_call_and_return_conditional_losses_4847695e4в1
*в'
!К
inputs         А
p 
к "-в*
#К 
tensor_0         А
Ъ З
)__inference_dropout_layer_call_fn_4847673Z4в1
*в'
!К
inputs         А
p
к ""К
unknown         АЗ
)__inference_dropout_layer_call_fn_4847678Z4в1
*в'
!К
inputs         А
p 
к ""К
unknown         АЄ
@__inference_gnn_layer_call_and_return_conditional_losses_4846519нъвц
╬в╩
╟к├
7
dst_connection_to_ipК
dst_connection_to_ip	
7
dst_ip_to_connectionК
dst_ip_to_connection	
3
feature_connectionК
feature_connection

n_cК	
n_c 	

n_iК	
n_i 	
7
src_connection_to_ipК
src_connection_to_ip	
7
src_ip_to_connectionК
src_ip_to_connection	
к

trainingp",в)
"К
tensor_0         
Ъ Є
@__inference_gnn_layer_call_and_return_conditional_losses_4847270нъвц
╬в╩
╟к├
7
dst_connection_to_ipК
dst_connection_to_ip	
7
dst_ip_to_connectionК
dst_ip_to_connection	
3
feature_connectionК
feature_connection

n_cК	
n_c 	

n_iК	
n_i 	
7
src_connection_to_ipК
src_connection_to_ip	
7
src_ip_to_connectionК
src_ip_to_connection	
к

trainingp ",в)
"К
tensor_0         
Ъ ╠
%__inference_gnn_layer_call_fn_4847313въвц
╬в╩
╟к├
7
dst_connection_to_ipК
dst_connection_to_ip	
7
dst_ip_to_connectionК
dst_ip_to_connection	
3
feature_connectionК
feature_connection

n_cК	
n_c 	

n_iК	
n_i 	
7
src_connection_to_ipК
src_connection_to_ip	
7
src_ip_to_connectionК
src_ip_to_connection	
к

trainingp"!К
unknown         ╠
%__inference_gnn_layer_call_fn_4847356въвц
╬в╩
╟к├
7
dst_connection_to_ipК
dst_connection_to_ip	
7
dst_ip_to_connectionК
dst_ip_to_connection	
3
feature_connectionК
feature_connection

n_cК	
n_c 	

n_iК	
n_i 	
7
src_connection_to_ipК
src_connection_to_ip	
7
src_ip_to_connectionК
src_ip_to_connection	
к

trainingp "!К
unknown         ╗
I__inference_sequential_1_layer_call_and_return_conditional_losses_4845526n9в6
/в,
"К
input_2         А
p

 
к "-в*
#К 
tensor_0         А
Ъ ╗
I__inference_sequential_1_layer_call_and_return_conditional_losses_4845541n9в6
/в,
"К
input_2         А
p 

 
к "-в*
#К 
tensor_0         А
Ъ Х
.__inference_sequential_1_layer_call_fn_4845550c9в6
/в,
"К
input_2         А
p

 
к ""К
unknown         АХ
.__inference_sequential_1_layer_call_fn_4845559c9в6
/в,
"К
input_2         А
p 

 
к ""К
unknown         А╛
I__inference_sequential_2_layer_call_and_return_conditional_losses_4845656q9в6
/в,
"К
input_3         А
p

 
к ",в)
"К
tensor_0         
Ъ ╛
I__inference_sequential_2_layer_call_and_return_conditional_losses_4845687q9в6
/в,
"К
input_3         А
p 

 
к ",в)
"К
tensor_0         
Ъ Ш
.__inference_sequential_2_layer_call_fn_4845704f9в6
/в,
"К
input_3         А
p

 
к "!К
unknown         Ш
.__inference_sequential_2_layer_call_fn_4845721f9в6
/в,
"К
input_3         А
p 

 
к "!К
unknown         ╣
G__inference_sequential_layer_call_and_return_conditional_losses_4845441n9в6
/в,
"К
input_1         А
p

 
к "-в*
#К 
tensor_0         А
Ъ ╣
G__inference_sequential_layer_call_and_return_conditional_losses_4845456n9в6
/в,
"К
input_1         А
p 

 
к "-в*
#К 
tensor_0         А
Ъ У
,__inference_sequential_layer_call_fn_4845465c9в6
/в,
"К
input_1         А
p

 
к ""К
unknown         АУ
,__inference_sequential_layer_call_fn_4845474c9в6
/в,
"К
input_1         А
p 

 
к ""К
unknown         А╟
%__inference_signature_wrapper_4847456Э╙в╧
в 
╟к├
7
dst_connection_to_ipК
dst_connection_to_ip	
7
dst_ip_to_connectionК
dst_ip_to_connection	
3
feature_connectionК
feature_connection

n_cК	
n_c 	

n_iК	
n_i 	
7
src_connection_to_ipК
src_connection_to_ip	
7
src_ip_to_connectionК
src_ip_to_connection	"3к0
.
output_1"К
output_1         д
N__inference_update_connection_layer_call_and_return_conditional_losses_4847629╤fвc
\вY
!К
inputs         А
0Ъ-
+К(
states_0                  
p
к "bв_
XвU
%К"

tensor_0_0         А
,Ъ)
'К$
tensor_0_1_0         А
Ъ д
N__inference_update_connection_layer_call_and_return_conditional_losses_4847668╤fвc
\вY
!К
inputs         А
0Ъ-
+К(
states_0                  
p 
к "bв_
XвU
%К"

tensor_0_0         А
,Ъ)
'К$
tensor_0_1_0         А
Ъ √
3__inference_update_connection_layer_call_fn_4847576├fвc
\вY
!К
inputs         А
0Ъ-
+К(
states_0                  
p
к "TвQ
#К 
tensor_0         А
*Ъ'
%К"

tensor_1_0         А√
3__inference_update_connection_layer_call_fn_4847590├fвc
\вY
!К
inputs         А
0Ъ-
+К(
states_0                  
p 
к "TвQ
#К 
tensor_0         А
*Ъ'
%К"

tensor_1_0         АФ
F__inference_update_ip_layer_call_and_return_conditional_losses_4847523╔^в[
TвQ
!К
inputs         А
(Ъ%
#К 
states_0         А
p
к "bв_
XвU
%К"

tensor_0_0         А
,Ъ)
'К$
tensor_0_1_0         А
Ъ Ф
F__inference_update_ip_layer_call_and_return_conditional_losses_4847562╔^в[
TвQ
!К
inputs         А
(Ъ%
#К 
states_0         А
p 
к "bв_
XвU
%К"

tensor_0_0         А
,Ъ)
'К$
tensor_0_1_0         А
Ъ ы
+__inference_update_ip_layer_call_fn_4847470╗^в[
TвQ
!К
inputs         А
(Ъ%
#К 
states_0         А
p
к "TвQ
#К 
tensor_0         А
*Ъ'
%К"

tensor_1_0         Аы
+__inference_update_ip_layer_call_fn_4847484╗^в[
TвQ
!К
inputs         А
(Ъ%
#К 
states_0         А
p 
к "TвQ
#К 
tensor_0         А
*Ъ'
%К"

tensor_1_0         А