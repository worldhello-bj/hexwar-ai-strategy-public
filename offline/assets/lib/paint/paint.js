$(function(){
	document.ondragstart=function(){return false;}//ie禁止拖拽
	document.onselectstart=function(){return false;}//ie禁止选定

	canvas1 = document.getElementById("paint");

	$(canvas1).bind('mousedown',function(event){
		if (event.which == 1) //鼠标左键
		{
			drawable=true;
			origin.x=event.clientX-canvas_offset.x;
			origin.y=event.clientY-canvas_offset.y;
			Ceju_ObjPos = GetID(event.clientX,event.clientY,4);
			var obj = $('#object div[pos='+Ceju_ObjPos+'][hide=0]').attr("id");
			Ceju_ObjID = (typeof(obj)=='undefined')?'':obj;
			Ceju_ObjE2 = (Ceju_ObjID)?parseInt($('#'+obj).attr('e2')):0;
		}
		if (event.which == 3) //鼠标右键
		{
			clear_paint();
		}
	});
	$(canvas1).bind('mouseup',function(event){
		if (event.which == 1) //鼠标左键
		{
			if(drawable==true) {
				drawable=false;
				context.clearRect(0,0,canvas_size.x,canvas_size.y);
				end.x=event.clientX-canvas_offset.x;
				end.y=event.clientY-canvas_offset.y;
				draw(context);
			} else {
				drawable=false;
			}
		}
	});
	$(canvas1).bind("mousemove",function(event){
		if(drawable==false)return;
		end.x=event.clientX-canvas_offset.x;
		end.y=event.clientY-canvas_offset.y;
		context.clearRect(0,0,canvas_size.x,canvas_size.y);
		draw(context);
	});

});

function draw(context){
	
	Ceju_TarPos = GetID(end.x,end.y,4);
	var tar = $('#object div[pos='+Ceju_TarPos+'][hide=0]').attr("id");
	Ceju_TarID = (typeof(tar)=='undefined')?'':tar;	
	Ceju_TarE2 = (Ceju_TarID)?parseInt($('#'+tar).attr('e2')):0;
	var Dist = Ceju(Ceju_ObjPos,Ceju_TarPos);
	var Gaocha = 0;
	Gaocha = Math.abs($('#'+Ceju_ObjPos).attr('elevation') - $('#'+Ceju_TarPos).attr('elevation'));
	var SeeStr = (See(Ceju_ObjPos,Ceju_TarPos,Dist,Ceju_ObjE2,Ceju_TarE2)==false)?'不通视':'通视';
	var CejuStr = '距离：'+Dist*GridMeter+'米（'+Dist+'格）';
		CejuStr += '高差：'+Gaocha+'米';
		CejuStr += '（'+SeeStr+'）';
	if (Ceju_ObjID && Ceju_TarID && SeeStr=='通视')
	{
		var GuanchaStr = (Observ(Ceju_ObjID,Ceju_TarID,Dist)==true)?'能观察':'不能观察';
		CejuStr += '（'+GuanchaStr+'）';
	}

	$('#map div').find('em').remove();

	if (SeeStopID)
	{
		var $obj = $('#'+SeeStopID);
		if ($obj.find('em').length==0)
		{
			$obj.append('<em class="text">'+$obj.attr('elevation')+'</em>');
		}
	}

	context.lineWidth = 10;
	context.strokeStyle = 'red';
	context.beginPath();
	context.moveTo(origin.x,origin.y);
	context.lineTo(end.x,end.y);
	context.stroke();

	context.lineWidth = 3;
	context.font = '600 18px 黑体';
	context.fillStyle = 'white';
	context.strokeStyle = 'red';
	context.strokeText(CejuStr,end.x,end.y-10);
	context.fillText(CejuStr,end.x,end.y-10);
}

function clear_paint() {
	context.clearRect(0,0,canvas_size.x,canvas_size.y);
	SeeStopID = '';
	ClearPath();
}

function paint_close() {
	clear_paint();
	$("#paint").hide();
}

function paintSet() {
	canvas1.width  = window.innerWidth;
	canvas1.height  = window.innerHeight;
	canvas_size={x:$("#paint").width(),y:$("#paint").height()};
	canvas_offset={x:$("#paint")[0].offsetLeft,y:$("#paint")[0].offsetTop};
	origin={x:0,y:0}
	end={x:0,y:0}
	type=1;
	drawable=false;
	context=canvas1.getContext('2d');
}