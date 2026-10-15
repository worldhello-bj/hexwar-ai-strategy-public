//CDNServer = '/lib/';
CDNServer = '/cdn1/';
GmapType = 'satellite'; //satellite terrain
DiedShow = true; //��ʾ���ߵ�λ
SecDiff = 1; //����ʱ��1���൱��ʵսʱ������� 1/(16/5)
SysMap = 0;
$Icon3d = true;
CommStat = false; //ͨ��״̬
RoomAI = 'BLUE'; //�˻��Կ���һ����AI
RoomXD = 7527; //��ǰ�붨

MoveCon = 0; //�Ƿ�һ������·����0��֧�֣�1֧�֣�
MoveConSec = 0; //���������м��ͣ��ʱ�䣨���룩
MoveInterval = 0; //������ʱ��ͣ��ʱ�䣨���룩

ObjClockCon = 1000; //��ѯ��������룩

AutoGetOff = false; //��ʼ̬�����ڳ��ϻ��ǳ���

aicommander = false; //�Ƿ��ύ�ɼ�����������true��ʾ�ύ����Բƽ̨��'AI'��ʾ�ύ��ѵ��ƽ̨
callbackurl = "http://cth.bkw100.com";

AttackLock = 1; //����ģʽ���ݿ�ʼǰ�ܷ������0���ܣ�1�ܣ�
AttackCon = 0; //AI����ӳٺ�����

CirArea = 0; //̬����ʾ������
ObjStackRule = 1; //�Ƿ����ѵ�����

GmapMoveX = ''; 
GmapMoveY = ''; 

JqueryDialogWidth = 480;
JqueryDialogHeight = 200;

CenterMoveX = 10;
CenterMoveY = 40000;

//�ж������
var Sys = {};
var ua = navigator.userAgent.toLowerCase();
var s;
(s = ua.match(/msie ([\d.]+)/)) ? Sys.ie = s[1] :
(s = ua.match(/firefox\/([\d.]+)/)) ? Sys.firefox = s[1] :
(s = ua.match(/chrome\/([\d.]+)/)) ? Sys.chrome = s[1] :
(s = ua.match(/opera.([\d.]+)/)) ? Sys.opera = s[1] :
(s = ua.match(/version\/([\d.]+).*safari/)) ? Sys.safari = s[1] : 0;

//�����������ѹ����
Ctrl = false;
document.onkeydown=function(e){
	e=window.event||e;
	switch(e.keyCode){
		case 37: //���
			SetCenter(CenterID-CenterMoveX); //������ż������������޷�����
			break;
		case 38: //���ϼ�
			SetCenter(CenterID-CenterMoveY);
			break;
		case 39: //�Ҽ�
			SetCenter(CenterID+CenterMoveX); //������ż������������޷�����
			break;
		case 40: //���¼�
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
		case 32: //�ո�
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