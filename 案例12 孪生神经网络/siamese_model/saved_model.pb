??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
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
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8??
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:@*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:@*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@0*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:@0*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:0*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d/kernel/m
?
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:@*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_1/kernel/m
?
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@0*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:@0*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:0*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d/kernel/v
?
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:@*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_1/kernel/v
?
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@0*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

:@0*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:0*
dtype0

NoOpNoOp
?K
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?K
value?KB?K B?K
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-1
layer-10
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
 
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer-7
layer_with_weights-2
layer-8
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
 	variables
!trainable_variables
"	keras_api
R
#regularization_losses
$	variables
%trainable_variables
&	keras_api
R
'regularization_losses
(	variables
)trainable_variables
*	keras_api
R
+regularization_losses
,	variables
-trainable_variables
.	keras_api
R
/regularization_losses
0	variables
1trainable_variables
2	keras_api
R
3regularization_losses
4	variables
5trainable_variables
6	keras_api
R
7regularization_losses
8	variables
9trainable_variables
:	keras_api
h

;kernel
<bias
=regularization_losses
>	variables
?trainable_variables
@	keras_api
?
Aiter

Bbeta_1

Cbeta_2
	Ddecay
Elearning_rate;m?<m?Fm?Gm?Hm?Im?Jm?Km?;v?<v?Fv?Gv?Hv?Iv?Jv?Kv?
 
8
F0
G1
H2
I3
J4
K5
;6
<7
8
F0
G1
H2
I3
J4
K5
;6
<7
?
regularization_losses
Lmetrics

Mlayers
	variables
trainable_variables
Nlayer_metrics
Olayer_regularization_losses
Pnon_trainable_variables
 
 
h

Fkernel
Gbias
Qregularization_losses
R	variables
Strainable_variables
T	keras_api
R
Uregularization_losses
V	variables
Wtrainable_variables
X	keras_api
R
Yregularization_losses
Z	variables
[trainable_variables
\	keras_api
h

Hkernel
Ibias
]regularization_losses
^	variables
_trainable_variables
`	keras_api
R
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
R
eregularization_losses
f	variables
gtrainable_variables
h	keras_api
R
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
h

Jkernel
Kbias
mregularization_losses
n	variables
otrainable_variables
p	keras_api
 
*
F0
G1
H2
I3
J4
K5
*
F0
G1
H2
I3
J4
K5
?
regularization_losses
qmetrics

rlayers
	variables
trainable_variables
slayer_metrics
tlayer_regularization_losses
unon_trainable_variables
 
 
 
?
regularization_losses
vmetrics

wlayers
 	variables
!trainable_variables
xlayer_metrics
ylayer_regularization_losses
znon_trainable_variables
 
 
 
?
#regularization_losses
{metrics

|layers
$	variables
%trainable_variables
}layer_metrics
~layer_regularization_losses
non_trainable_variables
 
 
 
?
'regularization_losses
?metrics
?layers
(	variables
)trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
+regularization_losses
?metrics
?layers
,	variables
-trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
/regularization_losses
?metrics
?layers
0	variables
1trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
3regularization_losses
?metrics
?layers
4	variables
5trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
7regularization_losses
?metrics
?layers
8	variables
9trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

;0
<1

;0
<1
?
=regularization_losses
?metrics
?layers
>	variables
?trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEconv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUE
dense/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
N
0
1
2
3
4
5
6
7
	8

9
10
 
 
 
 

F0
G1

F0
G1
?
Qregularization_losses
?metrics
?layers
R	variables
Strainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
Uregularization_losses
?metrics
?layers
V	variables
Wtrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
Yregularization_losses
?metrics
?layers
Z	variables
[trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 

H0
I1

H0
I1
?
]regularization_losses
?metrics
?layers
^	variables
_trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
aregularization_losses
?metrics
?layers
b	variables
ctrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
eregularization_losses
?metrics
?layers
f	variables
gtrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
 
 
?
iregularization_losses
?metrics
?layers
j	variables
ktrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 

J0
K1

J0
K1
?
mregularization_losses
?metrics
?layers
n	variables
otrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
 
?
0
1
2
3
4
5
6
7
8
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/conv2d/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEAdam/dense/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/conv2d/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEAdam/dense/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
serving_default_input_2Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
2
*
Tout
2*'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

GPU

CPU2*0J 8*-
f(R&
$__inference_signature_wrapper_446597
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*(
f#R!
__inference__traced_save_447211
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biastotalcounttotal_1count_1Adam/dense_1/kernel/mAdam/dense_1/bias/mAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/dense/kernel/vAdam/dense/bias/v*-
Tin&
$2"*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*+
f&R$
"__inference__traced_restore_447322??

?
P
4__inference_tf_op_layer_Maximum_layer_call_fn_446959

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Maximum_layer_call_and_return_conditional_losses_4463322
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_dense_1_layer_call_and_return_conditional_losses_447002

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?

?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_445979

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@:::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?#
?
A__inference_model_layer_call_and_return_conditional_losses_446118
input_3
conv2d_446018
conv2d_446020
conv2d_1_446054
conv2d_1_446056
dense_446112
dense_446114
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_446018conv2d_446020*
Tin
2*
Tout
2*/
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_4459452 
conv2d/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_4459612
max_pooling2d/PartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_4460362!
dropout/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv2d_1_446054conv2d_1_446056*
Tin
2*
Tout
2*/
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_4459792"
 conv2d_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4459952!
max_pooling2d_1/PartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_4460722#
!dropout_1/StatefulPartitionedCall?
(global_average_pooling2d/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_4460082*
(global_average_pooling2d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_446112dense_446114*
Tin
2*
Tout
2*'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4461012
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
U
9__inference_global_average_pooling2d_layer_call_fn_446014

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_4460082
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
j
N__inference_tf_op_layer_Square_layer_call_and_return_conditional_losses_446304

inputs
identityc
SquareSquareinputs*
T0*
_cloned(*'
_output_shapes
:?????????02
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????0:O K
'
_output_shapes
:?????????0
 
_user_specified_nameinputs
?

?
B__inference_conv2d_layer_call_and_return_conditional_losses_445945

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????:::i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
A__inference_dense_layer_call_and_return_conditional_losses_447075

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@0*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
(__inference_model_1_layer_call_fn_446785
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_4464872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?M
?
__inference__traced_save_447211
file_prefix-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_2e49c7ef1d014726a20dcb2b55c124d0/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*?
value?B?!B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 */
dtypes%
#2!	2
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::: : : : : :@:@:@@:@:@0:0: : : : :::@:@:@@:@:@0:0:::@:@:@@:@:@0:0: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 	

_output_shapes
:@:,
(
&
_output_shapes
:@@: 

_output_shapes
:@:$ 

_output_shapes

:@0: 

_output_shapes
:0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:$ 

_output_shapes

:@0: 

_output_shapes
:0:$ 

_output_shapes

:: 

_output_shapes
::,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:$  

_output_shapes

:@0: !

_output_shapes
:0:"

_output_shapes
: 
?
?
(__inference_model_1_layer_call_fn_446565
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_4465462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1:XT
/
_output_shapes
:?????????
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?
b
C__inference_dropout_layer_call_and_return_conditional_losses_447023

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
m
Q__inference_tf_op_layer_Maximum_1_layer_call_and_return_conditional_losses_446976

inputs
identity_
Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Maximum_1/y?
	Maximum_1MaximuminputsMaximum_1/y:output:0*
T0*
_cloned(*'
_output_shapes
:?????????2
	Maximum_1a
IdentityIdentityMaximum_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_447055

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_model_1_layer_call_fn_446807
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_4465462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?
M
1__inference_tf_op_layer_Sqrt_layer_call_fn_446991

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Sqrt_layer_call_and_return_conditional_losses_4463732
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
h
L__inference_tf_op_layer_Sqrt_layer_call_and_return_conditional_losses_446986

inputs
identity]
SqrtSqrtinputs*
T0*
_cloned(*'
_output_shapes
:?????????2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_446898

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*'
_output_shapes
:?????????0*(
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4461692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_447050

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
C__inference_model_1_layer_call_and_return_conditional_losses_446694
inputs_0
inputs_1/
+model_conv2d_conv2d_readvariableop_resource0
,model_conv2d_biasadd_readvariableop_resource1
-model_conv2d_1_conv2d_readvariableop_resource2
.model_conv2d_1_biasadd_readvariableop_resource.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"model/conv2d/Conv2D/ReadVariableOp?
model/conv2d/Conv2DConv2Dinputs_0*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
model/conv2d/Conv2D?
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#model/conv2d/BiasAdd/ReadVariableOp?
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
model/conv2d/BiasAdd?
model/conv2d/ReluRelumodel/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
model/conv2d/Relu?
model/max_pooling2d/MaxPoolMaxPoolmodel/conv2d/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
model/max_pooling2d/MaxPool
model/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
model/dropout/dropout/Const?
model/dropout/dropout/MulMul$model/max_pooling2d/MaxPool:output:0$model/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
model/dropout/dropout/Mul?
model/dropout/dropout/ShapeShape$model/max_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:2
model/dropout/dropout/Shape?
2model/dropout/dropout/random_uniform/RandomUniformRandomUniform$model/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype024
2model/dropout/dropout/random_uniform/RandomUniform?
$model/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2&
$model/dropout/dropout/GreaterEqual/y?
"model/dropout/dropout/GreaterEqualGreaterEqual;model/dropout/dropout/random_uniform/RandomUniform:output:0-model/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2$
"model/dropout/dropout/GreaterEqual?
model/dropout/dropout/CastCast&model/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
model/dropout/dropout/Cast?
model/dropout/dropout/Mul_1Mulmodel/dropout/dropout/Mul:z:0model/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
model/dropout/dropout/Mul_1?
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02&
$model/conv2d_1/Conv2D/ReadVariableOp?
model/conv2d_1/Conv2DConv2Dmodel/dropout/dropout/Mul_1:z:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
model/conv2d_1/Conv2D?
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv2d_1/BiasAdd/ReadVariableOp?
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
model/conv2d_1/BiasAdd?
model/conv2d_1/ReluRelumodel/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
model/conv2d_1/Relu?
model/max_pooling2d_1/MaxPoolMaxPool!model/conv2d_1/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
model/max_pooling2d_1/MaxPool?
model/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
model/dropout_1/dropout/Const?
model/dropout_1/dropout/MulMul&model/max_pooling2d_1/MaxPool:output:0&model/dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
model/dropout_1/dropout/Mul?
model/dropout_1/dropout/ShapeShape&model/max_pooling2d_1/MaxPool:output:0*
T0*
_output_shapes
:2
model/dropout_1/dropout/Shape?
4model/dropout_1/dropout/random_uniform/RandomUniformRandomUniform&model/dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype026
4model/dropout_1/dropout/random_uniform/RandomUniform?
&model/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2(
&model/dropout_1/dropout/GreaterEqual/y?
$model/dropout_1/dropout/GreaterEqualGreaterEqual=model/dropout_1/dropout/random_uniform/RandomUniform:output:0/model/dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2&
$model/dropout_1/dropout/GreaterEqual?
model/dropout_1/dropout/CastCast(model/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
model/dropout_1/dropout/Cast?
model/dropout_1/dropout/Mul_1Mulmodel/dropout_1/dropout/Mul:z:0 model/dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
model/dropout_1/dropout/Mul_1?
5model/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      27
5model/global_average_pooling2d/Mean/reduction_indices?
#model/global_average_pooling2d/MeanMean!model/dropout_1/dropout/Mul_1:z:0>model/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2%
#model/global_average_pooling2d/Mean?
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:@0*
dtype02#
!model/dense/MatMul/ReadVariableOp?
model/dense/MatMulMatMul,model/global_average_pooling2d/Mean:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model/dense/MatMul?
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02$
"model/dense/BiasAdd/ReadVariableOp?
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model/dense/BiasAdd?
$model_1/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02&
$model_1/conv2d/Conv2D/ReadVariableOp?
model_1/conv2d/Conv2DConv2Dinputs_1,model_1/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
model_1/conv2d/Conv2D?
%model_1/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model_1/conv2d/BiasAdd/ReadVariableOp?
model_1/conv2d/BiasAddBiasAddmodel_1/conv2d/Conv2D:output:0-model_1/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
model_1/conv2d/BiasAdd?
model_1/conv2d/ReluRelumodel_1/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
model_1/conv2d/Relu?
model_1/max_pooling2d/MaxPoolMaxPool!model_1/conv2d/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
model_1/max_pooling2d/MaxPool?
model_1/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
model_1/dropout/dropout/Const?
model_1/dropout/dropout/MulMul&model_1/max_pooling2d/MaxPool:output:0&model_1/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
model_1/dropout/dropout/Mul?
model_1/dropout/dropout/ShapeShape&model_1/max_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:2
model_1/dropout/dropout/Shape?
4model_1/dropout/dropout/random_uniform/RandomUniformRandomUniform&model_1/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype026
4model_1/dropout/dropout/random_uniform/RandomUniform?
&model_1/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2(
&model_1/dropout/dropout/GreaterEqual/y?
$model_1/dropout/dropout/GreaterEqualGreaterEqual=model_1/dropout/dropout/random_uniform/RandomUniform:output:0/model_1/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2&
$model_1/dropout/dropout/GreaterEqual?
model_1/dropout/dropout/CastCast(model_1/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
model_1/dropout/dropout/Cast?
model_1/dropout/dropout/Mul_1Mulmodel_1/dropout/dropout/Mul:z:0 model_1/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
model_1/dropout/dropout/Mul_1?
&model_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02(
&model_1/conv2d_1/Conv2D/ReadVariableOp?
model_1/conv2d_1/Conv2DConv2D!model_1/dropout/dropout/Mul_1:z:0.model_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
model_1/conv2d_1/Conv2D?
'model_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_1/conv2d_1/BiasAdd/ReadVariableOp?
model_1/conv2d_1/BiasAddBiasAdd model_1/conv2d_1/Conv2D:output:0/model_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
model_1/conv2d_1/BiasAdd?
model_1/conv2d_1/ReluRelu!model_1/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
model_1/conv2d_1/Relu?
model_1/max_pooling2d_1/MaxPoolMaxPool#model_1/conv2d_1/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2!
model_1/max_pooling2d_1/MaxPool?
model_1/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2!
model_1/dropout_1/dropout/Const?
model_1/dropout_1/dropout/MulMul(model_1/max_pooling2d_1/MaxPool:output:0(model_1/dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
model_1/dropout_1/dropout/Mul?
model_1/dropout_1/dropout/ShapeShape(model_1/max_pooling2d_1/MaxPool:output:0*
T0*
_output_shapes
:2!
model_1/dropout_1/dropout/Shape?
6model_1/dropout_1/dropout/random_uniform/RandomUniformRandomUniform(model_1/dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype028
6model_1/dropout_1/dropout/random_uniform/RandomUniform?
(model_1/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2*
(model_1/dropout_1/dropout/GreaterEqual/y?
&model_1/dropout_1/dropout/GreaterEqualGreaterEqual?model_1/dropout_1/dropout/random_uniform/RandomUniform:output:01model_1/dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2(
&model_1/dropout_1/dropout/GreaterEqual?
model_1/dropout_1/dropout/CastCast*model_1/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2 
model_1/dropout_1/dropout/Cast?
model_1/dropout_1/dropout/Mul_1Mul!model_1/dropout_1/dropout/Mul:z:0"model_1/dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2!
model_1/dropout_1/dropout/Mul_1?
7model_1/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      29
7model_1/global_average_pooling2d/Mean/reduction_indices?
%model_1/global_average_pooling2d/MeanMean#model_1/dropout_1/dropout/Mul_1:z:0@model_1/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2'
%model_1/global_average_pooling2d/Mean?
#model_1/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:@0*
dtype02%
#model_1/dense/MatMul/ReadVariableOp?
model_1/dense/MatMulMatMul.model_1/global_average_pooling2d/Mean:output:0+model_1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model_1/dense/MatMul?
$model_1/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02&
$model_1/dense/BiasAdd/ReadVariableOp?
model_1/dense/BiasAddBiasAddmodel_1/dense/MatMul:product:0,model_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model_1/dense/BiasAdd?
tf_op_layer_Sub/SubSubmodel/dense/BiasAdd:output:0model_1/dense/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:?????????02
tf_op_layer_Sub/Sub?
tf_op_layer_Square/SquareSquaretf_op_layer_Sub/Sub:z:0*
T0*
_cloned(*'
_output_shapes
:?????????02
tf_op_layer_Square/Square?
%tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_Sum/Sum/reduction_indices?
tf_op_layer_Sum/SumSumtf_op_layer_Square/Square:y:0.tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:?????????*
	keep_dims(2
tf_op_layer_Sum/Sum?
tf_op_layer_Maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
tf_op_layer_Maximum/Maximum/y?
tf_op_layer_Maximum/MaximumMaximumtf_op_layer_Sum/Sum:output:0&tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*'
_output_shapes
:?????????2
tf_op_layer_Maximum/Maximum?
tf_op_layer_Minimum/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_Minimum/Minimum/y?
tf_op_layer_Minimum/MinimumMinimumtf_op_layer_Maximum/Maximum:z:0&tf_op_layer_Minimum/Minimum/y:output:0*
T0*
_cloned(*'
_output_shapes
:?????????2
tf_op_layer_Minimum/Minimum?
!tf_op_layer_Maximum_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!tf_op_layer_Maximum_1/Maximum_1/y?
tf_op_layer_Maximum_1/Maximum_1Maximumtf_op_layer_Minimum/Minimum:z:0*tf_op_layer_Maximum_1/Maximum_1/y:output:0*
T0*
_cloned(*'
_output_shapes
:?????????2!
tf_op_layer_Maximum_1/Maximum_1?
tf_op_layer_Sqrt/SqrtSqrt#tf_op_layer_Maximum_1/Maximum_1:z:0*
T0*
_cloned(*'
_output_shapes
:?????????2
tf_op_layer_Sqrt/Sqrt?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMultf_op_layer_Sqrt/Sqrt:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Sigmoidg
IdentityIdentitydense_1/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????:::::::::Y U
/
_output_shapes
:?????????
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?
k
O__inference_tf_op_layer_Minimum_layer_call_and_return_conditional_losses_446346

inputs
identity[
	Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
	Minimum/yz
MinimumMinimuminputsMinimum/y:output:0*
T0*
_cloned(*'
_output_shapes
:?????????2	
Minimum_
IdentityIdentityMinimum:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
?
A__inference_model_layer_call_and_return_conditional_losses_446142
input_3
conv2d_446121
conv2d_446123
conv2d_1_446128
conv2d_1_446130
dense_446136
dense_446138
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_446121conv2d_446123*
Tin
2*
Tout
2*/
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_4459452 
conv2d/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_4459612
max_pooling2d/PartitionedCall?
dropout/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_4460412
dropout/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv2d_1_446128conv2d_1_446130*
Tin
2*
Tout
2*/
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_4459792"
 conv2d_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4459952!
max_pooling2d_1/PartitionedCall?
dropout_1/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_4460772
dropout_1/PartitionedCall?
(global_average_pooling2d/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_4460082*
(global_average_pooling2d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_446136dense_446138*
Tin
2*
Tout
2*'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4461012
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
&__inference_model_layer_call_fn_446184
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*'
_output_shapes
:?????????0*(
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4461692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
h
L__inference_tf_op_layer_Sqrt_layer_call_and_return_conditional_losses_446373

inputs
identity]
SqrtSqrtinputs*
T0*
_cloned(*'
_output_shapes
:?????????2
Sqrt\
IdentityIdentitySqrt:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
k
O__inference_tf_op_layer_Minimum_layer_call_and_return_conditional_losses_446965

inputs
identity[
	Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
	Minimum/yz
MinimumMinimuminputsMinimum/y:output:0*
T0*
_cloned(*'
_output_shapes
:?????????2	
Minimum_
IdentityIdentityMinimum:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?)
?
C__inference_model_1_layer_call_and_return_conditional_losses_446409
input_1
input_2
model_446264
model_446266
model_446268
model_446270
model_446272
model_446274
dense_1_446403
dense_1_446405
identity??dense_1/StatefulPartitionedCall?model/StatefulPartitionedCall?model_1/StatefulPartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCallinput_1model_446264model_446266model_446268model_446270model_446272model_446274*
Tin
	2*
Tout
2*'
_output_shapes
:?????????0*(
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4461692
model/StatefulPartitionedCall?
model_1/StatefulPartitionedCallStatefulPartitionedCallinput_2model_446264model_446266model_446268model_446270model_446272model_446274*
Tin
	2*
Tout
2*'
_output_shapes
:?????????0*(
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4461692!
model_1/StatefulPartitionedCall?
tf_op_layer_Sub/PartitionedCallPartitionedCall&model/StatefulPartitionedCall:output:0(model_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_4462902!
tf_op_layer_Sub/PartitionedCall?
"tf_op_layer_Square/PartitionedCallPartitionedCall(tf_op_layer_Sub/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Square_layer_call_and_return_conditional_losses_4463042$
"tf_op_layer_Square/PartitionedCall?
tf_op_layer_Sum/PartitionedCallPartitionedCall+tf_op_layer_Square/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_4463182!
tf_op_layer_Sum/PartitionedCall?
#tf_op_layer_Maximum/PartitionedCallPartitionedCall(tf_op_layer_Sum/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Maximum_layer_call_and_return_conditional_losses_4463322%
#tf_op_layer_Maximum/PartitionedCall?
#tf_op_layer_Minimum/PartitionedCallPartitionedCall,tf_op_layer_Maximum/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Minimum_layer_call_and_return_conditional_losses_4463462%
#tf_op_layer_Minimum/PartitionedCall?
%tf_op_layer_Maximum_1/PartitionedCallPartitionedCall,tf_op_layer_Minimum/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Maximum_1_layer_call_and_return_conditional_losses_4463602'
%tf_op_layer_Maximum_1/PartitionedCall?
 tf_op_layer_Sqrt/PartitionedCallPartitionedCall.tf_op_layer_Maximum_1/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Sqrt_layer_call_and_return_conditional_losses_4463732"
 tf_op_layer_Sqrt/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall)tf_op_layer_Sqrt/PartitionedCall:output:0dense_1_446403dense_1_446405*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4463922!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1:XT
/
_output_shapes
:?????????
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_446072

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
a
(__inference_dropout_layer_call_fn_447033

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_4460362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
p
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_446008

inputs
identity?
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_446915

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*'
_output_shapes
:?????????0*(
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4462102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
{
&__inference_dense_layer_call_fn_447084

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4461012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?4
?
A__inference_model_layer_call_and_return_conditional_losses_446851

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d/Relu?
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPools
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/dropout/Const?
dropout/dropout/MulMulmax_pooling2d/MaxPool:output:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout/dropout/Mul|
dropout/dropout/ShapeShapemax_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout/dropout/Mul_1?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Ddropout/dropout/Mul_1:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/Relu?
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_1/dropout/Const?
dropout_1/dropout/MulMul max_pooling2d_1/MaxPool:output:0 dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout_1/dropout/Mul?
dropout_1/dropout/ShapeShape max_pooling2d_1/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout_1/dropout/Mul_1?
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indices?
global_average_pooling2d/MeanMeandropout_1/dropout/Mul_1:z:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2
global_average_pooling2d/Mean?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@0*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMul&global_average_pooling2d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
dense/BiasAddj
IdentityIdentitydense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????:::::::W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
~
)__inference_conv2d_1_layer_call_fn_445989

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_4459792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_445995

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
b
C__inference_dropout_layer_call_and_return_conditional_losses_446036

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?e
?
!__inference__wrapped_model_445933
input_1
input_27
3model_1_model_conv2d_conv2d_readvariableop_resource8
4model_1_model_conv2d_biasadd_readvariableop_resource9
5model_1_model_conv2d_1_conv2d_readvariableop_resource:
6model_1_model_conv2d_1_biasadd_readvariableop_resource6
2model_1_model_dense_matmul_readvariableop_resource7
3model_1_model_dense_biasadd_readvariableop_resource2
.model_1_dense_1_matmul_readvariableop_resource3
/model_1_dense_1_biasadd_readvariableop_resource
identity??
*model_1/model/conv2d/Conv2D/ReadVariableOpReadVariableOp3model_1_model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02,
*model_1/model/conv2d/Conv2D/ReadVariableOp?
model_1/model/conv2d/Conv2DConv2Dinput_12model_1/model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
model_1/model/conv2d/Conv2D?
+model_1/model/conv2d/BiasAdd/ReadVariableOpReadVariableOp4model_1_model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+model_1/model/conv2d/BiasAdd/ReadVariableOp?
model_1/model/conv2d/BiasAddBiasAdd$model_1/model/conv2d/Conv2D:output:03model_1/model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
model_1/model/conv2d/BiasAdd?
model_1/model/conv2d/ReluRelu%model_1/model/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
model_1/model/conv2d/Relu?
#model_1/model/max_pooling2d/MaxPoolMaxPool'model_1/model/conv2d/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2%
#model_1/model/max_pooling2d/MaxPool?
model_1/model/dropout/IdentityIdentity,model_1/model/max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2 
model_1/model/dropout/Identity?
,model_1/model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp5model_1_model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02.
,model_1/model/conv2d_1/Conv2D/ReadVariableOp?
model_1/model/conv2d_1/Conv2DConv2D'model_1/model/dropout/Identity:output:04model_1/model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
model_1/model/conv2d_1/Conv2D?
-model_1/model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6model_1_model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-model_1/model/conv2d_1/BiasAdd/ReadVariableOp?
model_1/model/conv2d_1/BiasAddBiasAdd&model_1/model/conv2d_1/Conv2D:output:05model_1/model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2 
model_1/model/conv2d_1/BiasAdd?
model_1/model/conv2d_1/ReluRelu'model_1/model/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
model_1/model/conv2d_1/Relu?
%model_1/model/max_pooling2d_1/MaxPoolMaxPool)model_1/model/conv2d_1/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2'
%model_1/model/max_pooling2d_1/MaxPool?
 model_1/model/dropout_1/IdentityIdentity.model_1/model/max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2"
 model_1/model/dropout_1/Identity?
=model_1/model/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2?
=model_1/model/global_average_pooling2d/Mean/reduction_indices?
+model_1/model/global_average_pooling2d/MeanMean)model_1/model/dropout_1/Identity:output:0Fmodel_1/model/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2-
+model_1/model/global_average_pooling2d/Mean?
)model_1/model/dense/MatMul/ReadVariableOpReadVariableOp2model_1_model_dense_matmul_readvariableop_resource*
_output_shapes

:@0*
dtype02+
)model_1/model/dense/MatMul/ReadVariableOp?
model_1/model/dense/MatMulMatMul4model_1/model/global_average_pooling2d/Mean:output:01model_1/model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model_1/model/dense/MatMul?
*model_1/model/dense/BiasAdd/ReadVariableOpReadVariableOp3model_1_model_dense_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02,
*model_1/model/dense/BiasAdd/ReadVariableOp?
model_1/model/dense/BiasAddBiasAdd$model_1/model/dense/MatMul:product:02model_1/model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model_1/model/dense/BiasAdd?
,model_1/model_1/conv2d/Conv2D/ReadVariableOpReadVariableOp3model_1_model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02.
,model_1/model_1/conv2d/Conv2D/ReadVariableOp?
model_1/model_1/conv2d/Conv2DConv2Dinput_24model_1/model_1/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
model_1/model_1/conv2d/Conv2D?
-model_1/model_1/conv2d/BiasAdd/ReadVariableOpReadVariableOp4model_1_model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-model_1/model_1/conv2d/BiasAdd/ReadVariableOp?
model_1/model_1/conv2d/BiasAddBiasAdd&model_1/model_1/conv2d/Conv2D:output:05model_1/model_1/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2 
model_1/model_1/conv2d/BiasAdd?
model_1/model_1/conv2d/ReluRelu'model_1/model_1/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
model_1/model_1/conv2d/Relu?
%model_1/model_1/max_pooling2d/MaxPoolMaxPool)model_1/model_1/conv2d/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2'
%model_1/model_1/max_pooling2d/MaxPool?
 model_1/model_1/dropout/IdentityIdentity.model_1/model_1/max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2"
 model_1/model_1/dropout/Identity?
.model_1/model_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp5model_1_model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype020
.model_1/model_1/conv2d_1/Conv2D/ReadVariableOp?
model_1/model_1/conv2d_1/Conv2DConv2D)model_1/model_1/dropout/Identity:output:06model_1/model_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2!
model_1/model_1/conv2d_1/Conv2D?
/model_1/model_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6model_1_model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/model_1/model_1/conv2d_1/BiasAdd/ReadVariableOp?
 model_1/model_1/conv2d_1/BiasAddBiasAdd(model_1/model_1/conv2d_1/Conv2D:output:07model_1/model_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2"
 model_1/model_1/conv2d_1/BiasAdd?
model_1/model_1/conv2d_1/ReluRelu)model_1/model_1/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
model_1/model_1/conv2d_1/Relu?
'model_1/model_1/max_pooling2d_1/MaxPoolMaxPool+model_1/model_1/conv2d_1/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2)
'model_1/model_1/max_pooling2d_1/MaxPool?
"model_1/model_1/dropout_1/IdentityIdentity0model_1/model_1/max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2$
"model_1/model_1/dropout_1/Identity?
?model_1/model_1/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2A
?model_1/model_1/global_average_pooling2d/Mean/reduction_indices?
-model_1/model_1/global_average_pooling2d/MeanMean+model_1/model_1/dropout_1/Identity:output:0Hmodel_1/model_1/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2/
-model_1/model_1/global_average_pooling2d/Mean?
+model_1/model_1/dense/MatMul/ReadVariableOpReadVariableOp2model_1_model_dense_matmul_readvariableop_resource*
_output_shapes

:@0*
dtype02-
+model_1/model_1/dense/MatMul/ReadVariableOp?
model_1/model_1/dense/MatMulMatMul6model_1/model_1/global_average_pooling2d/Mean:output:03model_1/model_1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model_1/model_1/dense/MatMul?
,model_1/model_1/dense/BiasAdd/ReadVariableOpReadVariableOp3model_1_model_dense_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02.
,model_1/model_1/dense/BiasAdd/ReadVariableOp?
model_1/model_1/dense/BiasAddBiasAdd&model_1/model_1/dense/MatMul:product:04model_1/model_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model_1/model_1/dense/BiasAdd?
model_1/tf_op_layer_Sub/SubSub$model_1/model/dense/BiasAdd:output:0&model_1/model_1/dense/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:?????????02
model_1/tf_op_layer_Sub/Sub?
!model_1/tf_op_layer_Square/SquareSquaremodel_1/tf_op_layer_Sub/Sub:z:0*
T0*
_cloned(*'
_output_shapes
:?????????02#
!model_1/tf_op_layer_Square/Square?
-model_1/tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2/
-model_1/tf_op_layer_Sum/Sum/reduction_indices?
model_1/tf_op_layer_Sum/SumSum%model_1/tf_op_layer_Square/Square:y:06model_1/tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:?????????*
	keep_dims(2
model_1/tf_op_layer_Sum/Sum?
%model_1/tf_op_layer_Maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32'
%model_1/tf_op_layer_Maximum/Maximum/y?
#model_1/tf_op_layer_Maximum/MaximumMaximum$model_1/tf_op_layer_Sum/Sum:output:0.model_1/tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*'
_output_shapes
:?????????2%
#model_1/tf_op_layer_Maximum/Maximum?
%model_1/tf_op_layer_Minimum/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2'
%model_1/tf_op_layer_Minimum/Minimum/y?
#model_1/tf_op_layer_Minimum/MinimumMinimum'model_1/tf_op_layer_Maximum/Maximum:z:0.model_1/tf_op_layer_Minimum/Minimum/y:output:0*
T0*
_cloned(*'
_output_shapes
:?????????2%
#model_1/tf_op_layer_Minimum/Minimum?
)model_1/tf_op_layer_Maximum_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)model_1/tf_op_layer_Maximum_1/Maximum_1/y?
'model_1/tf_op_layer_Maximum_1/Maximum_1Maximum'model_1/tf_op_layer_Minimum/Minimum:z:02model_1/tf_op_layer_Maximum_1/Maximum_1/y:output:0*
T0*
_cloned(*'
_output_shapes
:?????????2)
'model_1/tf_op_layer_Maximum_1/Maximum_1?
model_1/tf_op_layer_Sqrt/SqrtSqrt+model_1/tf_op_layer_Maximum_1/Maximum_1:z:0*
T0*
_cloned(*'
_output_shapes
:?????????2
model_1/tf_op_layer_Sqrt/Sqrt?
%model_1/dense_1/MatMul/ReadVariableOpReadVariableOp.model_1_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02'
%model_1/dense_1/MatMul/ReadVariableOp?
model_1/dense_1/MatMulMatMul!model_1/tf_op_layer_Sqrt/Sqrt:y:0-model_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_1/MatMul?
&model_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_1/dense_1/BiasAdd/ReadVariableOp?
model_1/dense_1/BiasAddBiasAdd model_1/dense_1/MatMul:product:0.model_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_1/BiasAdd?
model_1/dense_1/SigmoidSigmoid model_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_1/dense_1/Sigmoido
IdentityIdentitymodel_1/dense_1/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????:::::::::X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1:XT
/
_output_shapes
:?????????
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?Z
?
C__inference_model_1_layer_call_and_return_conditional_losses_446763
inputs_0
inputs_1/
+model_conv2d_conv2d_readvariableop_resource0
,model_conv2d_biasadd_readvariableop_resource1
-model_conv2d_1_conv2d_readvariableop_resource2
.model_conv2d_1_biasadd_readvariableop_resource.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity??
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02$
"model/conv2d/Conv2D/ReadVariableOp?
model/conv2d/Conv2DConv2Dinputs_0*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
model/conv2d/Conv2D?
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02%
#model/conv2d/BiasAdd/ReadVariableOp?
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
model/conv2d/BiasAdd?
model/conv2d/ReluRelumodel/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
model/conv2d/Relu?
model/max_pooling2d/MaxPoolMaxPoolmodel/conv2d/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
model/max_pooling2d/MaxPool?
model/dropout/IdentityIdentity$model/max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2
model/dropout/Identity?
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02&
$model/conv2d_1/Conv2D/ReadVariableOp?
model/conv2d_1/Conv2DConv2Dmodel/dropout/Identity:output:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
model/conv2d_1/Conv2D?
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model/conv2d_1/BiasAdd/ReadVariableOp?
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
model/conv2d_1/BiasAdd?
model/conv2d_1/ReluRelumodel/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
model/conv2d_1/Relu?
model/max_pooling2d_1/MaxPoolMaxPool!model/conv2d_1/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
model/max_pooling2d_1/MaxPool?
model/dropout_1/IdentityIdentity&model/max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2
model/dropout_1/Identity?
5model/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      27
5model/global_average_pooling2d/Mean/reduction_indices?
#model/global_average_pooling2d/MeanMean!model/dropout_1/Identity:output:0>model/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2%
#model/global_average_pooling2d/Mean?
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:@0*
dtype02#
!model/dense/MatMul/ReadVariableOp?
model/dense/MatMulMatMul,model/global_average_pooling2d/Mean:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model/dense/MatMul?
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02$
"model/dense/BiasAdd/ReadVariableOp?
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model/dense/BiasAdd?
$model_1/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02&
$model_1/conv2d/Conv2D/ReadVariableOp?
model_1/conv2d/Conv2DConv2Dinputs_1,model_1/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
model_1/conv2d/Conv2D?
%model_1/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%model_1/conv2d/BiasAdd/ReadVariableOp?
model_1/conv2d/BiasAddBiasAddmodel_1/conv2d/Conv2D:output:0-model_1/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
model_1/conv2d/BiasAdd?
model_1/conv2d/ReluRelumodel_1/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
model_1/conv2d/Relu?
model_1/max_pooling2d/MaxPoolMaxPool!model_1/conv2d/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
model_1/max_pooling2d/MaxPool?
model_1/dropout/IdentityIdentity&model_1/max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2
model_1/dropout/Identity?
&model_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02(
&model_1/conv2d_1/Conv2D/ReadVariableOp?
model_1/conv2d_1/Conv2DConv2D!model_1/dropout/Identity:output:0.model_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
model_1/conv2d_1/Conv2D?
'model_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_1/conv2d_1/BiasAdd/ReadVariableOp?
model_1/conv2d_1/BiasAddBiasAdd model_1/conv2d_1/Conv2D:output:0/model_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
model_1/conv2d_1/BiasAdd?
model_1/conv2d_1/ReluRelu!model_1/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
model_1/conv2d_1/Relu?
model_1/max_pooling2d_1/MaxPoolMaxPool#model_1/conv2d_1/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2!
model_1/max_pooling2d_1/MaxPool?
model_1/dropout_1/IdentityIdentity(model_1/max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2
model_1/dropout_1/Identity?
7model_1/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      29
7model_1/global_average_pooling2d/Mean/reduction_indices?
%model_1/global_average_pooling2d/MeanMean#model_1/dropout_1/Identity:output:0@model_1/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2'
%model_1/global_average_pooling2d/Mean?
#model_1/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

:@0*
dtype02%
#model_1/dense/MatMul/ReadVariableOp?
model_1/dense/MatMulMatMul.model_1/global_average_pooling2d/Mean:output:0+model_1/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model_1/dense/MatMul?
$model_1/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02&
$model_1/dense/BiasAdd/ReadVariableOp?
model_1/dense/BiasAddBiasAddmodel_1/dense/MatMul:product:0,model_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
model_1/dense/BiasAdd?
tf_op_layer_Sub/SubSubmodel/dense/BiasAdd:output:0model_1/dense/BiasAdd:output:0*
T0*
_cloned(*'
_output_shapes
:?????????02
tf_op_layer_Sub/Sub?
tf_op_layer_Square/SquareSquaretf_op_layer_Sub/Sub:z:0*
T0*
_cloned(*'
_output_shapes
:?????????02
tf_op_layer_Square/Square?
%tf_op_layer_Sum/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_Sum/Sum/reduction_indices?
tf_op_layer_Sum/SumSumtf_op_layer_Square/Square:y:0.tf_op_layer_Sum/Sum/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:?????????*
	keep_dims(2
tf_op_layer_Sum/Sum?
tf_op_layer_Maximum/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
tf_op_layer_Maximum/Maximum/y?
tf_op_layer_Maximum/MaximumMaximumtf_op_layer_Sum/Sum:output:0&tf_op_layer_Maximum/Maximum/y:output:0*
T0*
_cloned(*'
_output_shapes
:?????????2
tf_op_layer_Maximum/Maximum?
tf_op_layer_Minimum/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
tf_op_layer_Minimum/Minimum/y?
tf_op_layer_Minimum/MinimumMinimumtf_op_layer_Maximum/Maximum:z:0&tf_op_layer_Minimum/Minimum/y:output:0*
T0*
_cloned(*'
_output_shapes
:?????????2
tf_op_layer_Minimum/Minimum?
!tf_op_layer_Maximum_1/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!tf_op_layer_Maximum_1/Maximum_1/y?
tf_op_layer_Maximum_1/Maximum_1Maximumtf_op_layer_Minimum/Minimum:z:0*tf_op_layer_Maximum_1/Maximum_1/y:output:0*
T0*
_cloned(*'
_output_shapes
:?????????2!
tf_op_layer_Maximum_1/Maximum_1?
tf_op_layer_Sqrt/SqrtSqrt#tf_op_layer_Maximum_1/Maximum_1:z:0*
T0*
_cloned(*'
_output_shapes
:?????????2
tf_op_layer_Sqrt/Sqrt?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMultf_op_layer_Sqrt/Sqrt:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Sigmoidg
IdentityIdentitydense_1/Sigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????:::::::::Y U
/
_output_shapes
:?????????
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?
j
N__inference_tf_op_layer_Square_layer_call_and_return_conditional_losses_446932

inputs
identityc
SquareSquareinputs*
T0*
_cloned(*'
_output_shapes
:?????????02
Square^
IdentityIdentity
Square:y:0*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????0:O K
'
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
?
(__inference_model_1_layer_call_fn_446506
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_4464872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1:XT
/
_output_shapes
:?????????
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
ԍ
?
"__inference__traced_restore_447322
file_prefix#
assignvariableop_dense_1_kernel#
assignvariableop_1_dense_1_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate$
 assignvariableop_7_conv2d_kernel"
assignvariableop_8_conv2d_bias&
"assignvariableop_9_conv2d_1_kernel%
!assignvariableop_10_conv2d_1_bias$
 assignvariableop_11_dense_kernel"
assignvariableop_12_dense_bias
assignvariableop_13_total
assignvariableop_14_count
assignvariableop_15_total_1
assignvariableop_16_count_1-
)assignvariableop_17_adam_dense_1_kernel_m+
'assignvariableop_18_adam_dense_1_bias_m,
(assignvariableop_19_adam_conv2d_kernel_m*
&assignvariableop_20_adam_conv2d_bias_m.
*assignvariableop_21_adam_conv2d_1_kernel_m,
(assignvariableop_22_adam_conv2d_1_bias_m+
'assignvariableop_23_adam_dense_kernel_m)
%assignvariableop_24_adam_dense_bias_m-
)assignvariableop_25_adam_dense_1_kernel_v+
'assignvariableop_26_adam_dense_1_bias_v,
(assignvariableop_27_adam_conv2d_kernel_v*
&assignvariableop_28_adam_conv2d_bias_v.
*assignvariableop_29_adam_conv2d_1_kernel_v,
(assignvariableop_30_adam_conv2d_1_bias_v+
'assignvariableop_31_adam_dense_kernel_v)
%assignvariableop_32_adam_dense_bias_v
identity_34??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*?
value?B?!B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::*/
dtypes%
#2!	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_1_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_kernelIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_conv2d_biasIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp"assignvariableop_9_conv2d_1_kernelIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp!assignvariableop_10_conv2d_1_biasIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_kernelIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_dense_biasIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_1_kernel_mIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_1_bias_mIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_conv2d_kernel_mIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_conv2d_bias_mIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv2d_1_kernel_mIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv2d_1_bias_mIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dense_kernel_mIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_dense_bias_mIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_1_kernel_vIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_1_bias_vIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_conv2d_kernel_vIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_conv2d_bias_vIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv2d_1_kernel_vIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv2d_1_bias_vIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_dense_kernel_vIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp%assignvariableop_32_adam_dense_bias_vIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_33?
Identity_34IdentityIdentity_33:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_34"#
identity_34Identity_34:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
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
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: 
?
k
O__inference_tf_op_layer_Maximum_layer_call_and_return_conditional_losses_446332

inputs
identity[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
	Maximum/yz
MaximumMaximuminputsMaximum/y:output:0*
T0*
_cloned(*'
_output_shapes
:?????????2	
Maximum_
IdentityIdentityMaximum:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_447028

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
F
*__inference_dropout_1_layer_call_fn_447065

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_4460772
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
J
.__inference_max_pooling2d_layer_call_fn_445967

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_4459612
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
&__inference_model_layer_call_fn_446225
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*'
_output_shapes
:?????????0*(
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4462102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_3:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
m
Q__inference_tf_op_layer_Maximum_1_layer_call_and_return_conditional_losses_446360

inputs
identity_
Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Maximum_1/y?
	Maximum_1MaximuminputsMaximum_1/y:output:0*
T0*
_cloned(*'
_output_shapes
:?????????2
	Maximum_1a
IdentityIdentityMaximum_1:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
k
O__inference_tf_op_layer_Maximum_layer_call_and_return_conditional_losses_446954

inputs
identity[
	Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
	Maximum/yz
MaximumMaximuminputsMaximum/y:output:0*
T0*
_cloned(*'
_output_shapes
:?????????2	
Maximum_
IdentityIdentityMaximum:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_445961

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?)
?
C__inference_model_1_layer_call_and_return_conditional_losses_446446
input_1
input_2
model_446413
model_446415
model_446417
model_446419
model_446421
model_446423
dense_1_446440
dense_1_446442
identity??dense_1/StatefulPartitionedCall?model/StatefulPartitionedCall?model_1/StatefulPartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCallinput_1model_446413model_446415model_446417model_446419model_446421model_446423*
Tin
	2*
Tout
2*'
_output_shapes
:?????????0*(
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4462102
model/StatefulPartitionedCall?
model_1/StatefulPartitionedCallStatefulPartitionedCallinput_2model_446413model_446415model_446417model_446419model_446421model_446423*
Tin
	2*
Tout
2*'
_output_shapes
:?????????0*(
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4462102!
model_1/StatefulPartitionedCall?
tf_op_layer_Sub/PartitionedCallPartitionedCall&model/StatefulPartitionedCall:output:0(model_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_4462902!
tf_op_layer_Sub/PartitionedCall?
"tf_op_layer_Square/PartitionedCallPartitionedCall(tf_op_layer_Sub/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Square_layer_call_and_return_conditional_losses_4463042$
"tf_op_layer_Square/PartitionedCall?
tf_op_layer_Sum/PartitionedCallPartitionedCall+tf_op_layer_Square/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_4463182!
tf_op_layer_Sum/PartitionedCall?
#tf_op_layer_Maximum/PartitionedCallPartitionedCall(tf_op_layer_Sum/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Maximum_layer_call_and_return_conditional_losses_4463322%
#tf_op_layer_Maximum/PartitionedCall?
#tf_op_layer_Minimum/PartitionedCallPartitionedCall,tf_op_layer_Maximum/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Minimum_layer_call_and_return_conditional_losses_4463462%
#tf_op_layer_Minimum/PartitionedCall?
%tf_op_layer_Maximum_1/PartitionedCallPartitionedCall,tf_op_layer_Minimum/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Maximum_1_layer_call_and_return_conditional_losses_4463602'
%tf_op_layer_Maximum_1/PartitionedCall?
 tf_op_layer_Sqrt/PartitionedCallPartitionedCall.tf_op_layer_Maximum_1/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Sqrt_layer_call_and_return_conditional_losses_4463732"
 tf_op_layer_Sqrt/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall)tf_op_layer_Sqrt/PartitionedCall:output:0dense_1_446440dense_1_446442*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4463922!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1:XT
/
_output_shapes
:?????????
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?
g
K__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_446943

inputs
identityp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSuminputsSum/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:?????????*
	keep_dims(2
Sum`
IdentityIdentitySum:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????0:O K
'
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_446041

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
R
6__inference_tf_op_layer_Maximum_1_layer_call_fn_446981

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Maximum_1_layer_call_and_return_conditional_losses_4463602
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_1_layer_call_fn_446001

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4459952
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?)
?
C__inference_model_1_layer_call_and_return_conditional_losses_446487

inputs
inputs_1
model_446454
model_446456
model_446458
model_446460
model_446462
model_446464
dense_1_446481
dense_1_446483
identity??dense_1/StatefulPartitionedCall?model/StatefulPartitionedCall?model_1/StatefulPartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_446454model_446456model_446458model_446460model_446462model_446464*
Tin
	2*
Tout
2*'
_output_shapes
:?????????0*(
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4461692
model/StatefulPartitionedCall?
model_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1model_446454model_446456model_446458model_446460model_446462model_446464*
Tin
	2*
Tout
2*'
_output_shapes
:?????????0*(
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4461692!
model_1/StatefulPartitionedCall?
tf_op_layer_Sub/PartitionedCallPartitionedCall&model/StatefulPartitionedCall:output:0(model_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_4462902!
tf_op_layer_Sub/PartitionedCall?
"tf_op_layer_Square/PartitionedCallPartitionedCall(tf_op_layer_Sub/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Square_layer_call_and_return_conditional_losses_4463042$
"tf_op_layer_Square/PartitionedCall?
tf_op_layer_Sum/PartitionedCallPartitionedCall+tf_op_layer_Square/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_4463182!
tf_op_layer_Sum/PartitionedCall?
#tf_op_layer_Maximum/PartitionedCallPartitionedCall(tf_op_layer_Sum/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Maximum_layer_call_and_return_conditional_losses_4463322%
#tf_op_layer_Maximum/PartitionedCall?
#tf_op_layer_Minimum/PartitionedCallPartitionedCall,tf_op_layer_Maximum/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Minimum_layer_call_and_return_conditional_losses_4463462%
#tf_op_layer_Minimum/PartitionedCall?
%tf_op_layer_Maximum_1/PartitionedCallPartitionedCall,tf_op_layer_Minimum/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Maximum_1_layer_call_and_return_conditional_losses_4463602'
%tf_op_layer_Maximum_1/PartitionedCall?
 tf_op_layer_Sqrt/PartitionedCallPartitionedCall.tf_op_layer_Maximum_1/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Sqrt_layer_call_and_return_conditional_losses_4463732"
 tf_op_layer_Sqrt/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall)tf_op_layer_Sqrt/PartitionedCall:output:0dense_1_446481dense_1_446483*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4463922!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?!
?
A__inference_model_layer_call_and_return_conditional_losses_446881

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d/Relu?
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
dropout/IdentityIdentitymax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2
dropout/Identity?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Ddropout/Identity:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/Relu?
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
dropout_1/IdentityIdentity max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????@2
dropout_1/Identity?
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indices?
global_average_pooling2d/MeanMeandropout_1/Identity:output:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????@2
global_average_pooling2d/Mean?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@0*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMul&global_average_pooling2d/Mean:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
dense/BiasAddj
IdentityIdentitydense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????:::::::W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
}
(__inference_dense_1_layer_call_fn_447011

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4463922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
L
0__inference_tf_op_layer_Sum_layer_call_fn_446948

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_4463182
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????0:O K
'
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
w
K__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_446921
inputs_0
inputs_1
identityf
SubSubinputs_0inputs_1*
T0*
_cloned(*'
_output_shapes
:?????????02
Sub[
IdentityIdentitySub:z:0*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????0:?????????0:Q M
'
_output_shapes
:?????????0
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????0
"
_user_specified_name
inputs/1
?
?
C__inference_dense_1_layer_call_and_return_conditional_losses_446392

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
P
4__inference_tf_op_layer_Minimum_layer_call_fn_446970

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Minimum_layer_call_and_return_conditional_losses_4463462
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
?
A__inference_model_layer_call_and_return_conditional_losses_446210

inputs
conv2d_446189
conv2d_446191
conv2d_1_446196
conv2d_1_446198
dense_446204
dense_446206
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_446189conv2d_446191*
Tin
2*
Tout
2*/
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_4459452 
conv2d/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_4459612
max_pooling2d/PartitionedCall?
dropout/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_4460412
dropout/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv2d_1_446196conv2d_1_446198*
Tin
2*
Tout
2*/
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_4459792"
 conv2d_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4459952!
max_pooling2d_1/PartitionedCall?
dropout_1/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_4460772
dropout_1/PartitionedCall?
(global_average_pooling2d/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_4460082*
(global_average_pooling2d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_446204dense_446206*
Tin
2*
Tout
2*'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4461012
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_446077

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
$__inference_signature_wrapper_446597
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

GPU

CPU2*0J 8**
f%R#
!__inference__wrapped_model_4459332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1:XT
/
_output_shapes
:?????????
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?
?
A__inference_dense_layer_call_and_return_conditional_losses_446101

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@0*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????02	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
u
K__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_446290

inputs
inputs_1
identityd
SubSubinputsinputs_1*
T0*
_cloned(*'
_output_shapes
:?????????02
Sub[
IdentityIdentitySub:z:0*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????0:?????????0:O K
'
_output_shapes
:?????????0
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????0
 
_user_specified_nameinputs
?)
?
C__inference_model_1_layer_call_and_return_conditional_losses_446546

inputs
inputs_1
model_446513
model_446515
model_446517
model_446519
model_446521
model_446523
dense_1_446540
dense_1_446542
identity??dense_1/StatefulPartitionedCall?model/StatefulPartitionedCall?model_1/StatefulPartitionedCall?
model/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_446513model_446515model_446517model_446519model_446521model_446523*
Tin
	2*
Tout
2*'
_output_shapes
:?????????0*(
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4462102
model/StatefulPartitionedCall?
model_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1model_446513model_446515model_446517model_446519model_446521model_446523*
Tin
	2*
Tout
2*'
_output_shapes
:?????????0*(
_read_only_resource_inputs

*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_model_layer_call_and_return_conditional_losses_4462102!
model_1/StatefulPartitionedCall?
tf_op_layer_Sub/PartitionedCallPartitionedCall&model/StatefulPartitionedCall:output:0(model_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_4462902!
tf_op_layer_Sub/PartitionedCall?
"tf_op_layer_Square/PartitionedCallPartitionedCall(tf_op_layer_Sub/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Square_layer_call_and_return_conditional_losses_4463042$
"tf_op_layer_Square/PartitionedCall?
tf_op_layer_Sum/PartitionedCallPartitionedCall+tf_op_layer_Square/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_4463182!
tf_op_layer_Sum/PartitionedCall?
#tf_op_layer_Maximum/PartitionedCallPartitionedCall(tf_op_layer_Sum/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Maximum_layer_call_and_return_conditional_losses_4463322%
#tf_op_layer_Maximum/PartitionedCall?
#tf_op_layer_Minimum/PartitionedCallPartitionedCall,tf_op_layer_Maximum/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*X
fSRQ
O__inference_tf_op_layer_Minimum_layer_call_and_return_conditional_losses_4463462%
#tf_op_layer_Minimum/PartitionedCall?
%tf_op_layer_Maximum_1/PartitionedCallPartitionedCall,tf_op_layer_Minimum/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*Z
fURS
Q__inference_tf_op_layer_Maximum_1_layer_call_and_return_conditional_losses_4463602'
%tf_op_layer_Maximum_1/PartitionedCall?
 tf_op_layer_Sqrt/PartitionedCallPartitionedCall.tf_op_layer_Maximum_1/PartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*U
fPRN
L__inference_tf_op_layer_Sqrt_layer_call_and_return_conditional_losses_4463732"
 tf_op_layer_Sqrt/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall)tf_op_layer_Sqrt/PartitionedCall:output:0dense_1_446540dense_1_446542*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4463922!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
?
g
K__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_446318

inputs
identityp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSuminputsSum/reduction_indices:output:0*
T0*
_cloned(*'
_output_shapes
:?????????*
	keep_dims(2
Sum`
IdentityIdentitySum:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????0:O K
'
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
O
3__inference_tf_op_layer_Square_layer_call_fn_446937

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*'
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*W
fRRP
N__inference_tf_op_layer_Square_layer_call_and_return_conditional_losses_4463042
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????0:O K
'
_output_shapes
:?????????0
 
_user_specified_nameinputs
?
\
0__inference_tf_op_layer_Sub_layer_call_fn_446927
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*'
_output_shapes
:?????????0* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_4462902
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????0:?????????0:Q M
'
_output_shapes
:?????????0
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????0
"
_user_specified_name
inputs/1
?
c
*__inference_dropout_1_layer_call_fn_447060

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_4460722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
|
'__inference_conv2d_layer_call_fn_445955

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_4459452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?#
?
A__inference_model_layer_call_and_return_conditional_losses_446169

inputs
conv2d_446148
conv2d_446150
conv2d_1_446155
conv2d_1_446157
dense_446163
dense_446165
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_446148conv2d_446150*
Tin
2*
Tout
2*/
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_4459452 
conv2d/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_4459612
max_pooling2d/PartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_4460362!
dropout/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv2d_1_446155conv2d_1_446157*
Tin
2*
Tout
2*/
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_4459792"
 conv2d_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4459952!
max_pooling2d_1/PartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_4460722#
!dropout_1/StatefulPartitionedCall?
(global_average_pooling2d/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*'
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*]
fXRV
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_4460082*
(global_average_pooling2d/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0dense_446163dense_446165*
Tin
2*
Tout
2*'
_output_shapes
:?????????0*$
_read_only_resource_inputs
*-
config_proto

GPU

CPU2*0J 8*J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_4461012
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????02

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
D
(__inference_dropout_layer_call_fn_447038

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

GPU

CPU2*0J 8*L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_4460412
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????
C
input_28
serving_default_input_2:0?????????;
dense_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?z
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-1
layer-10
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?w
_tf_keras_model?w{"class_name": "Model", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 48, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["global_average_pooling2d", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["dense", 0, 0]]}, "name": "model", "inbound_nodes": [[["input_1", 0, 0, {}]], [["input_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sub", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub", "op": "Sub", "input": ["model/Identity", "model_1/Identity"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Sub", "inbound_nodes": [[["model", 1, 0, {}], ["model", 2, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Square", "trainable": true, "dtype": "float32", "node_def": {"name": "Square", "op": "Square", "input": ["Sub"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Square", "inbound_nodes": [[["tf_op_layer_Sub", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sum", "trainable": true, "dtype": "float32", "node_def": {"name": "Sum", "op": "Sum", "input": ["Square", "Sum/reduction_indices"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "keep_dims": {"b": true}}}, "constants": {"1": 1}}, "name": "tf_op_layer_Sum", "inbound_nodes": [[["tf_op_layer_Square", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Maximum", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum", "op": "Maximum", "input": ["Sum", "Maximum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0000000116860974e-07}}, "name": "tf_op_layer_Maximum", "inbound_nodes": [[["tf_op_layer_Sum", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Minimum", "trainable": true, "dtype": "float32", "node_def": {"name": "Minimum", "op": "Minimum", "input": ["Maximum", "Minimum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": Infinity}}, "name": "tf_op_layer_Minimum", "inbound_nodes": [[["tf_op_layer_Maximum", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Maximum_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum_1", "op": "Maximum", "input": ["Minimum", "Maximum_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 0.0}}, "name": "tf_op_layer_Maximum_1", "inbound_nodes": [[["tf_op_layer_Minimum", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sqrt", "trainable": true, "dtype": "float32", "node_def": {"name": "Sqrt", "op": "Sqrt", "input": ["Maximum_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Sqrt", "inbound_nodes": [[["tf_op_layer_Maximum_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["tf_op_layer_Sqrt", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["dense_1", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 28, 28, 1]}, {"class_name": "TensorShape", "items": [null, 28, 28, 1]}], "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 48, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["global_average_pooling2d", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["dense", 0, 0]]}, "name": "model", "inbound_nodes": [[["input_1", 0, 0, {}]], [["input_2", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sub", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub", "op": "Sub", "input": ["model/Identity", "model_1/Identity"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Sub", "inbound_nodes": [[["model", 1, 0, {}], ["model", 2, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Square", "trainable": true, "dtype": "float32", "node_def": {"name": "Square", "op": "Square", "input": ["Sub"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Square", "inbound_nodes": [[["tf_op_layer_Sub", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sum", "trainable": true, "dtype": "float32", "node_def": {"name": "Sum", "op": "Sum", "input": ["Square", "Sum/reduction_indices"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "keep_dims": {"b": true}}}, "constants": {"1": 1}}, "name": "tf_op_layer_Sum", "inbound_nodes": [[["tf_op_layer_Square", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Maximum", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum", "op": "Maximum", "input": ["Sum", "Maximum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0000000116860974e-07}}, "name": "tf_op_layer_Maximum", "inbound_nodes": [[["tf_op_layer_Sum", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Minimum", "trainable": true, "dtype": "float32", "node_def": {"name": "Minimum", "op": "Minimum", "input": ["Maximum", "Minimum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": Infinity}}, "name": "tf_op_layer_Minimum", "inbound_nodes": [[["tf_op_layer_Maximum", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Maximum_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum_1", "op": "Maximum", "input": ["Minimum", "Maximum_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 0.0}}, "name": "tf_op_layer_Maximum_1", "inbound_nodes": [[["tf_op_layer_Minimum", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Sqrt", "trainable": true, "dtype": "float32", "node_def": {"name": "Sqrt", "op": "Sqrt", "input": ["Maximum_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}, "name": "tf_op_layer_Sqrt", "inbound_nodes": [[["tf_op_layer_Maximum_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["tf_op_layer_Sqrt", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["dense_1", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
?>
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer-7
layer_with_weights-2
layer-8
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?<
_tf_keras_model?;{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 48, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["global_average_pooling2d", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["dense", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 48, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["global_average_pooling2d", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["dense", 0, 0]]}}}
?
regularization_losses
 	variables
!trainable_variables
"	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Sub", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Sub", "trainable": true, "dtype": "float32", "node_def": {"name": "Sub", "op": "Sub", "input": ["model/Identity", "model_1/Identity"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?
#regularization_losses
$	variables
%trainable_variables
&	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Square", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Square", "trainable": true, "dtype": "float32", "node_def": {"name": "Square", "op": "Square", "input": ["Sub"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?
'regularization_losses
(	variables
)trainable_variables
*	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Sum", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Sum", "trainable": true, "dtype": "float32", "node_def": {"name": "Sum", "op": "Sum", "input": ["Square", "Sum/reduction_indices"], "attr": {"Tidx": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}, "keep_dims": {"b": true}}}, "constants": {"1": 1}}}
?
+regularization_losses
,	variables
-trainable_variables
.	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Maximum", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Maximum", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum", "op": "Maximum", "input": ["Sum", "Maximum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 1.0000000116860974e-07}}}
?
/regularization_losses
0	variables
1trainable_variables
2	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Minimum", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Minimum", "trainable": true, "dtype": "float32", "node_def": {"name": "Minimum", "op": "Minimum", "input": ["Maximum", "Minimum/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": Infinity}}}
?
3regularization_losses
4	variables
5trainable_variables
6	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Maximum_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Maximum_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Maximum_1", "op": "Maximum", "input": ["Minimum", "Maximum_1/y"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {"1": 0.0}}}
?
7regularization_losses
8	variables
9trainable_variables
:	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Sqrt", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "Sqrt", "trainable": true, "dtype": "float32", "node_def": {"name": "Sqrt", "op": "Sqrt", "input": ["Maximum_1"], "attr": {"T": {"type": "DT_FLOAT"}}}, "constants": {}}}
?

;kernel
<bias
=regularization_losses
>	variables
?trainable_variables
@	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?
Aiter

Bbeta_1

Cbeta_2
	Ddecay
Elearning_rate;m?<m?Fm?Gm?Hm?Im?Jm?Km?;v?<v?Fv?Gv?Hv?Iv?Jv?Kv?"
	optimizer
 "
trackable_list_wrapper
X
F0
G1
H2
I3
J4
K5
;6
<7"
trackable_list_wrapper
X
F0
G1
H2
I3
J4
K5
;6
<7"
trackable_list_wrapper
?
regularization_losses
Lmetrics

Mlayers
	variables
trainable_variables
Nlayer_metrics
Olayer_regularization_losses
Pnon_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
?	

Fkernel
Gbias
Qregularization_losses
R	variables
Strainable_variables
T	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}
?
Uregularization_losses
V	variables
Wtrainable_variables
X	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
Yregularization_losses
Z	variables
[trainable_variables
\	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
?	

Hkernel
Ibias
]regularization_losses
^	variables
_trainable_variables
`	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 64]}}
?
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
eregularization_losses
f	variables
gtrainable_variables
h	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
?
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GlobalAveragePooling2D", "name": "global_average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?

Jkernel
Kbias
mregularization_losses
n	variables
otrainable_variables
p	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 48, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
 "
trackable_list_wrapper
J
F0
G1
H2
I3
J4
K5"
trackable_list_wrapper
J
F0
G1
H2
I3
J4
K5"
trackable_list_wrapper
?
regularization_losses
qmetrics

rlayers
	variables
trainable_variables
slayer_metrics
tlayer_regularization_losses
unon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
vmetrics

wlayers
 	variables
!trainable_variables
xlayer_metrics
ylayer_regularization_losses
znon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
#regularization_losses
{metrics

|layers
$	variables
%trainable_variables
}layer_metrics
~layer_regularization_losses
non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
'regularization_losses
?metrics
?layers
(	variables
)trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
+regularization_losses
?metrics
?layers
,	variables
-trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
/regularization_losses
?metrics
?layers
0	variables
1trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
3regularization_losses
?metrics
?layers
4	variables
5trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
7regularization_losses
?metrics
?layers
8	variables
9trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :2dense_1/kernel
:2dense_1/bias
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
?
=regularization_losses
?metrics
?layers
>	variables
?trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
':%@2conv2d/kernel
:@2conv2d/bias
):'@@2conv2d_1/kernel
:@2conv2d_1/bias
:@02dense/kernel
:02
dense/bias
0
?0
?1"
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
?
Qregularization_losses
?metrics
?layers
R	variables
Strainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Uregularization_losses
?metrics
?layers
V	variables
Wtrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Yregularization_losses
?metrics
?layers
Z	variables
[trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
?
]regularization_losses
?metrics
?layers
^	variables
_trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
aregularization_losses
?metrics
?layers
b	variables
ctrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
eregularization_losses
?metrics
?layers
f	variables
gtrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
iregularization_losses
?metrics
?layers
j	variables
ktrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
?
mregularization_losses
?metrics
?layers
n	variables
otrainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
8"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
%:#2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
,:*@2Adam/conv2d/kernel/m
:@2Adam/conv2d/bias/m
.:,@@2Adam/conv2d_1/kernel/m
 :@2Adam/conv2d_1/bias/m
#:!@02Adam/dense/kernel/m
:02Adam/dense/bias/m
%:#2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
,:*@2Adam/conv2d/kernel/v
:@2Adam/conv2d/bias/v
.:,@@2Adam/conv2d_1/kernel/v
 :@2Adam/conv2d_1/bias/v
#:!@02Adam/dense/kernel/v
:02Adam/dense/bias/v
?2?
!__inference__wrapped_model_445933?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *^?[
Y?V
)?&
input_1?????????
)?&
input_2?????????
?2?
C__inference_model_1_layer_call_and_return_conditional_losses_446694
C__inference_model_1_layer_call_and_return_conditional_losses_446446
C__inference_model_1_layer_call_and_return_conditional_losses_446409
C__inference_model_1_layer_call_and_return_conditional_losses_446763?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_model_1_layer_call_fn_446785
(__inference_model_1_layer_call_fn_446506
(__inference_model_1_layer_call_fn_446807
(__inference_model_1_layer_call_fn_446565?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_model_layer_call_and_return_conditional_losses_446881
A__inference_model_layer_call_and_return_conditional_losses_446142
A__inference_model_layer_call_and_return_conditional_losses_446851
A__inference_model_layer_call_and_return_conditional_losses_446118?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_model_layer_call_fn_446915
&__inference_model_layer_call_fn_446184
&__inference_model_layer_call_fn_446898
&__inference_model_layer_call_fn_446225?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_446921?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_tf_op_layer_Sub_layer_call_fn_446927?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_tf_op_layer_Square_layer_call_and_return_conditional_losses_446932?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_tf_op_layer_Square_layer_call_fn_446937?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_446943?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_tf_op_layer_Sum_layer_call_fn_446948?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_tf_op_layer_Maximum_layer_call_and_return_conditional_losses_446954?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_tf_op_layer_Maximum_layer_call_fn_446959?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_tf_op_layer_Minimum_layer_call_and_return_conditional_losses_446965?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_tf_op_layer_Minimum_layer_call_fn_446970?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_tf_op_layer_Maximum_1_layer_call_and_return_conditional_losses_446976?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
6__inference_tf_op_layer_Maximum_1_layer_call_fn_446981?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_tf_op_layer_Sqrt_layer_call_and_return_conditional_losses_446986?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_tf_op_layer_Sqrt_layer_call_fn_446991?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_1_layer_call_and_return_conditional_losses_447002?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_1_layer_call_fn_447011?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:B8
$__inference_signature_wrapper_446597input_1input_2
?2?
B__inference_conv2d_layer_call_and_return_conditional_losses_445945?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
'__inference_conv2d_layer_call_fn_445955?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_445961?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
.__inference_max_pooling2d_layer_call_fn_445967?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
C__inference_dropout_layer_call_and_return_conditional_losses_447023
C__inference_dropout_layer_call_and_return_conditional_losses_447028?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_dropout_layer_call_fn_447038
(__inference_dropout_layer_call_fn_447033?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_445979?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
)__inference_conv2d_1_layer_call_fn_445989?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_445995?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
0__inference_max_pooling2d_1_layer_call_fn_446001?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
E__inference_dropout_1_layer_call_and_return_conditional_losses_447050
E__inference_dropout_1_layer_call_and_return_conditional_losses_447055?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_dropout_1_layer_call_fn_447060
*__inference_dropout_1_layer_call_fn_447065?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_446008?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
9__inference_global_average_pooling2d_layer_call_fn_446014?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
A__inference_dense_layer_call_and_return_conditional_losses_447075?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_layer_call_fn_447084?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_445933?FGHIJK;<h?e
^?[
Y?V
)?&
input_1?????????
)?&
input_2?????????
? "1?.
,
dense_1!?
dense_1??????????
D__inference_conv2d_1_layer_call_and_return_conditional_losses_445979?HII?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
)__inference_conv2d_1_layer_call_fn_445989?HII?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
B__inference_conv2d_layer_call_and_return_conditional_losses_445945?FGI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
'__inference_conv2d_layer_call_fn_445955?FGI?F
??<
:?7
inputs+???????????????????????????
? "2?/+???????????????????????????@?
C__inference_dense_1_layer_call_and_return_conditional_losses_447002\;</?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
(__inference_dense_1_layer_call_fn_447011O;</?,
%?"
 ?
inputs?????????
? "???????????
A__inference_dense_layer_call_and_return_conditional_losses_447075\JK/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????0
? y
&__inference_dense_layer_call_fn_447084OJK/?,
%?"
 ?
inputs?????????@
? "??????????0?
E__inference_dropout_1_layer_call_and_return_conditional_losses_447050l;?8
1?.
(?%
inputs?????????@
p
? "-?*
#? 
0?????????@
? ?
E__inference_dropout_1_layer_call_and_return_conditional_losses_447055l;?8
1?.
(?%
inputs?????????@
p 
? "-?*
#? 
0?????????@
? ?
*__inference_dropout_1_layer_call_fn_447060_;?8
1?.
(?%
inputs?????????@
p
? " ??????????@?
*__inference_dropout_1_layer_call_fn_447065_;?8
1?.
(?%
inputs?????????@
p 
? " ??????????@?
C__inference_dropout_layer_call_and_return_conditional_losses_447023l;?8
1?.
(?%
inputs?????????@
p
? "-?*
#? 
0?????????@
? ?
C__inference_dropout_layer_call_and_return_conditional_losses_447028l;?8
1?.
(?%
inputs?????????@
p 
? "-?*
#? 
0?????????@
? ?
(__inference_dropout_layer_call_fn_447033_;?8
1?.
(?%
inputs?????????@
p
? " ??????????@?
(__inference_dropout_layer_call_fn_447038_;?8
1?.
(?%
inputs?????????@
p 
? " ??????????@?
T__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_446008?R?O
H?E
C?@
inputs4????????????????????????????????????
? ".?+
$?!
0??????????????????
? ?
9__inference_global_average_pooling2d_layer_call_fn_446014wR?O
H?E
C?@
inputs4????????????????????????????????????
? "!????????????????????
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_445995?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_1_layer_call_fn_446001?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_445961?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_max_pooling2d_layer_call_fn_445967?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
C__inference_model_1_layer_call_and_return_conditional_losses_446409?FGHIJK;<p?m
f?c
Y?V
)?&
input_1?????????
)?&
input_2?????????
p

 
? "%?"
?
0?????????
? ?
C__inference_model_1_layer_call_and_return_conditional_losses_446446?FGHIJK;<p?m
f?c
Y?V
)?&
input_1?????????
)?&
input_2?????????
p 

 
? "%?"
?
0?????????
? ?
C__inference_model_1_layer_call_and_return_conditional_losses_446694?FGHIJK;<r?o
h?e
[?X
*?'
inputs/0?????????
*?'
inputs/1?????????
p

 
? "%?"
?
0?????????
? ?
C__inference_model_1_layer_call_and_return_conditional_losses_446763?FGHIJK;<r?o
h?e
[?X
*?'
inputs/0?????????
*?'
inputs/1?????????
p 

 
? "%?"
?
0?????????
? ?
(__inference_model_1_layer_call_fn_446506?FGHIJK;<p?m
f?c
Y?V
)?&
input_1?????????
)?&
input_2?????????
p

 
? "???????????
(__inference_model_1_layer_call_fn_446565?FGHIJK;<p?m
f?c
Y?V
)?&
input_1?????????
)?&
input_2?????????
p 

 
? "???????????
(__inference_model_1_layer_call_fn_446785?FGHIJK;<r?o
h?e
[?X
*?'
inputs/0?????????
*?'
inputs/1?????????
p

 
? "???????????
(__inference_model_1_layer_call_fn_446807?FGHIJK;<r?o
h?e
[?X
*?'
inputs/0?????????
*?'
inputs/1?????????
p 

 
? "???????????
A__inference_model_layer_call_and_return_conditional_losses_446118qFGHIJK@?=
6?3
)?&
input_3?????????
p

 
? "%?"
?
0?????????0
? ?
A__inference_model_layer_call_and_return_conditional_losses_446142qFGHIJK@?=
6?3
)?&
input_3?????????
p 

 
? "%?"
?
0?????????0
? ?
A__inference_model_layer_call_and_return_conditional_losses_446851pFGHIJK??<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0?????????0
? ?
A__inference_model_layer_call_and_return_conditional_losses_446881pFGHIJK??<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0?????????0
? ?
&__inference_model_layer_call_fn_446184dFGHIJK@?=
6?3
)?&
input_3?????????
p

 
? "??????????0?
&__inference_model_layer_call_fn_446225dFGHIJK@?=
6?3
)?&
input_3?????????
p 

 
? "??????????0?
&__inference_model_layer_call_fn_446898cFGHIJK??<
5?2
(?%
inputs?????????
p

 
? "??????????0?
&__inference_model_layer_call_fn_446915cFGHIJK??<
5?2
(?%
inputs?????????
p 

 
? "??????????0?
$__inference_signature_wrapper_446597?FGHIJK;<y?v
? 
o?l
4
input_1)?&
input_1?????????
4
input_2)?&
input_2?????????"1?.
,
dense_1!?
dense_1??????????
Q__inference_tf_op_layer_Maximum_1_layer_call_and_return_conditional_losses_446976X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
6__inference_tf_op_layer_Maximum_1_layer_call_fn_446981K/?,
%?"
 ?
inputs?????????
? "???????????
O__inference_tf_op_layer_Maximum_layer_call_and_return_conditional_losses_446954X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
4__inference_tf_op_layer_Maximum_layer_call_fn_446959K/?,
%?"
 ?
inputs?????????
? "???????????
O__inference_tf_op_layer_Minimum_layer_call_and_return_conditional_losses_446965X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
4__inference_tf_op_layer_Minimum_layer_call_fn_446970K/?,
%?"
 ?
inputs?????????
? "???????????
L__inference_tf_op_layer_Sqrt_layer_call_and_return_conditional_losses_446986X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
1__inference_tf_op_layer_Sqrt_layer_call_fn_446991K/?,
%?"
 ?
inputs?????????
? "???????????
N__inference_tf_op_layer_Square_layer_call_and_return_conditional_losses_446932X/?,
%?"
 ?
inputs?????????0
? "%?"
?
0?????????0
? ?
3__inference_tf_op_layer_Square_layer_call_fn_446937K/?,
%?"
 ?
inputs?????????0
? "??????????0?
K__inference_tf_op_layer_Sub_layer_call_and_return_conditional_losses_446921?Z?W
P?M
K?H
"?
inputs/0?????????0
"?
inputs/1?????????0
? "%?"
?
0?????????0
? ?
0__inference_tf_op_layer_Sub_layer_call_fn_446927vZ?W
P?M
K?H
"?
inputs/0?????????0
"?
inputs/1?????????0
? "??????????0?
K__inference_tf_op_layer_Sum_layer_call_and_return_conditional_losses_446943X/?,
%?"
 ?
inputs?????????0
? "%?"
?
0?????????
? 
0__inference_tf_op_layer_Sum_layer_call_fn_446948K/?,
%?"
 ?
inputs?????????0
? "??????????