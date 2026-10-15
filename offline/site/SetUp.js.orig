//CDNServer = '/lib/';
CDNServer = 'http://cdn1.hexwar.cn/';
GmapType = 'satellite'; //satellite terrain
DiedShow = true; //显示被歼单位
SecDiff = 1; //操作时间1秒相当于实战时间多少秒 1/(16/5)
SysMap = 0;
$Icon3d = true;
CommStat = false; //通信状态
RoomAI = 'BLUE'; //人机对抗哪一方是AI
RoomXD = 7527; //当前想定

MoveCon = 0; //是否一键生成路径（0不支持，1支持）
MoveConSec = 0; //连续机动中间的停顿时间（毫秒）
MoveInterval = 0; //逐格机动时的停顿时间（毫秒）

ObjClockCon = 1000; //轮询间隔（毫秒）

AutoGetOff = false; //初始态势人在车上还是车下

aicommander = false; //是否提交成绩到第三方，true表示提交到域圆平台，'AI'表示提交到训练平台
callbackurl = "http://cth.bkw100.com";

AttackLock = 1; //红蓝模式推演开始前能否射击（0不能，1能）
AttackCon = 0; //AI射击延迟毫秒数

CirArea = 0; //态势显示火力线
ObjStackRule = 1; //是否开启堆叠规则

GmapMoveX = ''; 
GmapMoveY = ''; 

JqueryDialogWidth = 480;
JqueryDialogHeight = 200;

CenterMoveX = 10;
CenterMoveY = 40000;

//判断浏览器
var Sys = {};
var ua = navigator.userAgent.toLowerCase();
var s;
(s = ua.match(/msie ([\d.]+)/)) ? Sys.ie = s[1] :
(s = ua.match(/firefox\/([\d.]+)/)) ? Sys.firefox = s[1] :
(s = ua.match(/chrome\/([\d.]+)/)) ? Sys.chrome = s[1] :
(s = ua.match(/opera.([\d.]+)/)) ? Sys.opera = s[1] :
(s = ua.match(/version\/([\d.]+).*safari/)) ? Sys.safari = s[1] : 0;

//方向键（不能压缩）
Ctrl = false;
document.onkeydown=function(e){
	e=window.event||e;
	switch(e.keyCode){
		case 37: //左键
			SetCenter(CenterID-CenterMoveX); //必须是偶数，否则方里格无法对齐
			break;
		case 38: //向上键
			SetCenter(CenterID-CenterMoveY);
			break;
		case 39: //右键
			SetCenter(CenterID+CenterMoveX); //必须是偶数，否则方里格无法对齐
			break;
		case 40: //向下键
			SetCenter(CenterID+CenterMoveY);
			break;
		case 17: //Ctrl
			Ctrl = true;
			break;
		default:
			break;
	}
}

document.onkeyup=function(e){
	e=window.event||e;
	switch(e.keyCode){
		case 17: //Ctrl
			Ctrl = false;
			break;
		case 32: //空格
			showmenuie5(MyObjID);
			break;
		case 67: //C
			PaintShow();
			break;
		case 68: //D
			AttackToggle(MyObjID);
			break;
		case 83: //S
			MoveToggle();
			break;
		default:
			break;
	}
}