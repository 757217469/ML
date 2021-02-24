import json

print(json.loads(
    '''{"eventBody":{"approval":null,"articleList":null,"atList":null,"atLists":null,"auditReason":null,"auditStatus":"AUDIT-01","author":"平安邯郸","category":null,"categoryList":null,"circle":null,"circleList":[],"city":null,"columnList":[{"columnCode":"1204343405560340481","columnPath":"1_1204343405560340481_","contentCode":"E_B_792281","createTime":1592795326144,"id":864478,"sortNum":0,"status":null,"updateTime":1592795326144}],"columnPath":"1_1204343405560340481_","commentSwitch":0,"content":null,"contentCode":"E_B_792281","contentId":792281,"contentType":"CT003","country":null,"countryCode":null,"coverType":"CVT01","coverUrl":"[\"http://flv3.people.com.cn/dev1/mvideo/images/2020/06/22/20200622_8241_kcos_rmtr_50_586.jpg\"]","coverUrlList":["http://flv3.people.com.cn/dev1/mvideo/images/2020/06/22/20200622_8241_kcos_rmtr_50_586.jpg"],"createTime":1592795326131,"executiveEditor":null,"flowId":null,"hiddenMap":null,"isPush":null,"isTiming":null,"latitude":null,"location":null,"longitude":null,"materialList":[{"contentCode":"E_B_792281","height":"0","id":null,"isCover":null,"materialCode":"","materialDesc":null,"materialSize":"0","materialType":4,"materialUrl":"http://flv3.people.com.cn/dev1/mvideo/vodfiles/2020/06/22/278d8c4502dbe2c4ff7590502a371169_c.mp4","sort":null,"sourceType":null,"status":null,"title":null,"videoPlayCount":null,"width":"0"}],"mediaId":null,"mediaType":"pgc","onePublish":"PRD002","outLink":null,"pgcId":147838,"posterUrl":"http://haikenews.static.haiwainet.cn/poster_v6/2020/0622/960512a0-b8a2-43d4-b9cd-0a4bc5c40aa8.jpg?id=E_B_792281&contentType=CT003","priority":0,"productCode":"PRD002","province":null,"publishStatus":"PUB-01","qrUrl":"http://haikenews.static.haiwainet.cn/qrcode_v6/content/2020/0622/f566878c-8e77-465a-b71e-9c74801eeb20.png","releaseTime":1592795326131,"shareUrl":"http://mk.haiwainet.cn/video/hkvideo.html?mid=E_B_792281&contentType=CT003","shortTitle":null,"shoulderTitle":null,"showType":null,"skuNo":null,"skuPrice":null,"sortOrder":null,"sourceCode":"155163","sourceType":"SUT01","status":null,"summary":"邯郸交巡警渚河大队民警在路口执勤时,发现一名小女孩独自走在非机动车道上，嘴里一直喊着“妈妈……妈妈……”民警想到这个孩子极有可能是与家人走失，找不到回家的路，又考虑到此路段来往车辆较多，具有极大的安全隐患。民警快步走到小女孩面前，将她带到路边，蹲下身拉着小女孩的手仔细地询问情况。","tag":null,"thirdContentId":"1_cbe179ac1afd54dc54d1b7ad98ded2b1","thirdId":"1","title":"走失女童徘徊街头 交警助其找到父母","transcodeStatus":null,"updateTime":1592795326131,"userId":null,"viceTitle":null},"eventHead":{"eventCode":"E-CT003-100","operation":"","storeCode":"C-PUBLISH"}}'''.replace(
        'null', 'None')))
s = {"eventBody": {"approval": None, "articleList": None, "atList": None, "atLists": None, "auditReason": None,
                   "auditStatus": "AUDIT-01", "author": "平安邯郸", "category": None, "categoryList": None, "circle": None,
                   "circleList": [], "city": None, "columnList": [
        {"columnCode": "1204343405560340481", "columnPath": "1_1204343405560340481_", "contentCode": "E_B_792281",
         "createTime": 1592795326144, "id": 864478, "sortNum": 0, "status": None, "updateTime": 1592795326144}],
                   "columnPath": "1_1204343405560340481_", "commentSwitch": 0, "content": None,
                   "contentCode": "E_B_792281", "contentId": 792281, "contentType": "CT003", "country": None,
                   "countryCode": None, "coverType": "CVT01", "coverUrl": [
        "http://flv3.people.com.cn/dev1/mvideo/images/2020/06/22/20200622_8241_kcos_rmtr_50_586.jpg"],
                   "coverUrlList": [
                       "http://flv3.people.com.cn/dev1/mvideo/images/2020/06/22/20200622_8241_kcos_rmtr_50_586.jpg"],
                   "createTime": 1592795326131, "executiveEditor": None, "flowId": None, "hiddenMap": None,
                   "isPush": None,
                   "isTiming": None, "latitude": None, "location": None, "longitude": None, "materialList": [
        {"contentCode": "E_B_792281", "height": "0", "id": None, "isCover": None, "materialCode": "",
         "materialDesc": None, "materialSize": "0", "materialType": 4,
         "materialUrl": "http://flv3.people.com.cn/dev1/mvideo/vodfiles/2020/06/22/278d8c4502dbe2c4ff7590502a371169_c.mp4",
         "sort": None, "sourceType": None, "status": None, "title": None, "videoPlayCount": None, "width": "0"}],
                   "mediaId": None, "mediaType": "pgc", "onePublish": "PRD002", "outLink": None, "pgcId": 147838,
                   "posterUrl": "http://haikenews.static.haiwainet.cn/poster_v6/2020/0622/960512a0-b8a2-43d4-b9cd-0a4bc5c40aa8.jpg?id=E_B_792281&contentType=CT003",
                   "priority": 0, "productCode": "PRD002", "province": None, "publishStatus": "PUB-01",
                   "qrUrl": "http://haikenews.static.haiwainet.cn/qrcode_v6/content/2020/0622/f566878c-8e77-465a-b71e-9c74801eeb20.png",
                   "releaseTime": 1592795326131,
                   "shareUrl": "http://mk.haiwainet.cn/video/hkvideo.html?mid=E_B_792281&contentType=CT003",
                   "shortTitle": None,
                   "shoulderTitle": None, "showType": None, "skuNo": None, "skuPrice": None, "sortOrder": None,
                   "sourceCode": "155163", "sourceType": "SUT01", "status": None,
                   "summary": "邯郸交巡警渚河大队民警在路口执勤时,发现一名小女孩独自走在非机动车道上，嘴里一直喊着“妈妈……妈妈……”民警想到这个孩子极有可能是与家人走失，找不到回家的路，又考虑到此路段来往车辆较多，具有极大的安全隐患。民警快步走到小女孩面前，将她带到路边，蹲下身拉着小女孩的手仔细地询问情况。",
                   "tag": None, "thirdContentId": "1_cbe179ac1afd54dc54d1b7ad98ded2b1", "thirdId": "1",
                   "title": "走失女童徘徊街头 交警助其找到父母",
                   "transcodeStatus": None, "updateTime": 1592795326131, "userId": None, "viceTitle": None},
     "eventHead": {"eventCode": "E-CT003-100", "operation": "", "storeCode": "C-PUBLISH"}}
