/*!
 * ScrambleTextPlugin 3.1.1
 * https://greensock.com
 *
 * @license Copyright 2020, GreenSock. All rights reserved.
 * Subject to the terms at https://greensock.com/standard-license or for Club GreenSock members, the agreement issued with that membership.
 * @author: Jack Doyle, jack@greensock.com
 */

!(function (D, u) {
    "object" == typeof exports && "undefined" != typeof module ? u(exports) : "function" == typeof define && define.amd ? define(["exports"], u) : u(((D = D || self).window = D.window || {}));
})(this, function (D) {
    "use strict";
    var n = /(^\s+|\s+$)/g,
        r = /([\uD800-\uDBFF][\uDC00-\uDFFF](?:[\u200D\uFE0F][\uD800-\uDBFF][\uDC00-\uDFFF]){2,}|\uD83D\uDC69(?:\u200D(?:(?:\uD83D\uDC69\u200D)?\uD83D\uDC67|(?:\uD83D\uDC69\u200D)?\uD83D\uDC66)|\uD83C[\uDFFB-\uDFFF])|\uD83D\uDC69\u200D(?:\uD83D\uDC69\u200D)?\uD83D\uDC66\u200D\uD83D\uDC66|\uD83D\uDC69\u200D(?:\uD83D\uDC69\u200D)?\uD83D\uDC67\u200D(?:\uD83D[\uDC66\uDC67])|\uD83C\uDFF3\uFE0F\u200D\uD83C\uDF08|(?:\uD83C[\uDFC3\uDFC4\uDFCA]|\uD83D[\uDC6E\uDC71\uDC73\uDC77\uDC81\uDC82\uDC86\uDC87\uDE45-\uDE47\uDE4B\uDE4D\uDE4E\uDEA3\uDEB4-\uDEB6]|\uD83E[\uDD26\uDD37-\uDD39\uDD3D\uDD3E\uDDD6-\uDDDD])(?:\uD83C[\uDFFB-\uDFFF])\u200D[\u2640\u2642]\uFE0F|\uD83D\uDC69(?:\uD83C[\uDFFB-\uDFFF])\u200D(?:\uD83C[\uDF3E\uDF73\uDF93\uDFA4\uDFA8\uDFEB\uDFED]|\uD83D[\uDCBB\uDCBC\uDD27\uDD2C\uDE80\uDE92])|(?:\uD83C[\uDFC3\uDFC4\uDFCA]|\uD83D[\uDC6E\uDC6F\uDC71\uDC73\uDC77\uDC81\uDC82\uDC86\uDC87\uDE45-\uDE47\uDE4B\uDE4D\uDE4E\uDEA3\uDEB4-\uDEB6]|\uD83E[\uDD26\uDD37-\uDD39\uDD3C-\uDD3E\uDDD6-\uDDDF])\u200D[\u2640\u2642]\uFE0F|\uD83C\uDDFD\uD83C\uDDF0|\uD83C\uDDF6\uD83C\uDDE6|\uD83C\uDDF4\uD83C\uDDF2|\uD83C\uDDE9(?:\uD83C[\uDDEA\uDDEC\uDDEF\uDDF0\uDDF2\uDDF4\uDDFF])|\uD83C\uDDF7(?:\uD83C[\uDDEA\uDDF4\uDDF8\uDDFA\uDDFC])|\uD83C\uDDE8(?:\uD83C[\uDDE6\uDDE8\uDDE9\uDDEB-\uDDEE\uDDF0-\uDDF5\uDDF7\uDDFA-\uDDFF])|(?:\u26F9|\uD83C[\uDFCB\uDFCC]|\uD83D\uDD75)(?:\uFE0F\u200D[\u2640\u2642]|(?:\uD83C[\uDFFB-\uDFFF])\u200D[\u2640\u2642])\uFE0F|(?:\uD83D\uDC41\uFE0F\u200D\uD83D\uDDE8|\uD83D\uDC69(?:\uD83C[\uDFFB-\uDFFF])\u200D[\u2695\u2696\u2708]|\uD83D\uDC69\u200D[\u2695\u2696\u2708]|\uD83D\uDC68(?:(?:\uD83C[\uDFFB-\uDFFF])\u200D[\u2695\u2696\u2708]|\u200D[\u2695\u2696\u2708]))\uFE0F|\uD83C\uDDF2(?:\uD83C[\uDDE6\uDDE8-\uDDED\uDDF0-\uDDFF])|\uD83D\uDC69\u200D(?:\uD83C[\uDF3E\uDF73\uDF93\uDFA4\uDFA8\uDFEB\uDFED]|\uD83D[\uDCBB\uDCBC\uDD27\uDD2C\uDE80\uDE92]|\u2764\uFE0F\u200D(?:\uD83D\uDC8B\u200D(?:\uD83D[\uDC68\uDC69])|\uD83D[\uDC68\uDC69]))|\uD83C\uDDF1(?:\uD83C[\uDDE6-\uDDE8\uDDEE\uDDF0\uDDF7-\uDDFB\uDDFE])|\uD83C\uDDEF(?:\uD83C[\uDDEA\uDDF2\uDDF4\uDDF5])|\uD83C\uDDED(?:\uD83C[\uDDF0\uDDF2\uDDF3\uDDF7\uDDF9\uDDFA])|\uD83C\uDDEB(?:\uD83C[\uDDEE-\uDDF0\uDDF2\uDDF4\uDDF7])|[#\*0-9]\uFE0F\u20E3|\uD83C\uDDE7(?:\uD83C[\uDDE6\uDDE7\uDDE9-\uDDEF\uDDF1-\uDDF4\uDDF6-\uDDF9\uDDFB\uDDFC\uDDFE\uDDFF])|\uD83C\uDDE6(?:\uD83C[\uDDE8-\uDDEC\uDDEE\uDDF1\uDDF2\uDDF4\uDDF6-\uDDFA\uDDFC\uDDFD\uDDFF])|\uD83C\uDDFF(?:\uD83C[\uDDE6\uDDF2\uDDFC])|\uD83C\uDDF5(?:\uD83C[\uDDE6\uDDEA-\uDDED\uDDF0-\uDDF3\uDDF7-\uDDF9\uDDFC\uDDFE])|\uD83C\uDDFB(?:\uD83C[\uDDE6\uDDE8\uDDEA\uDDEC\uDDEE\uDDF3\uDDFA])|\uD83C\uDDF3(?:\uD83C[\uDDE6\uDDE8\uDDEA-\uDDEC\uDDEE\uDDF1\uDDF4\uDDF5\uDDF7\uDDFA\uDDFF])|\uD83C\uDFF4\uDB40\uDC67\uDB40\uDC62(?:\uDB40\uDC77\uDB40\uDC6C\uDB40\uDC73|\uDB40\uDC73\uDB40\uDC63\uDB40\uDC74|\uDB40\uDC65\uDB40\uDC6E\uDB40\uDC67)\uDB40\uDC7F|\uD83D\uDC68(?:\u200D(?:\u2764\uFE0F\u200D(?:\uD83D\uDC8B\u200D)?\uD83D\uDC68|(?:(?:\uD83D[\uDC68\uDC69])\u200D)?\uD83D\uDC66\u200D\uD83D\uDC66|(?:(?:\uD83D[\uDC68\uDC69])\u200D)?\uD83D\uDC67\u200D(?:\uD83D[\uDC66\uDC67])|\uD83C[\uDF3E\uDF73\uDF93\uDFA4\uDFA8\uDFEB\uDFED]|\uD83D[\uDCBB\uDCBC\uDD27\uDD2C\uDE80\uDE92])|(?:\uD83C[\uDFFB-\uDFFF])\u200D(?:\uD83C[\uDF3E\uDF73\uDF93\uDFA4\uDFA8\uDFEB\uDFED]|\uD83D[\uDCBB\uDCBC\uDD27\uDD2C\uDE80\uDE92]))|\uD83C\uDDF8(?:\uD83C[\uDDE6-\uDDEA\uDDEC-\uDDF4\uDDF7-\uDDF9\uDDFB\uDDFD-\uDDFF])|\uD83C\uDDF0(?:\uD83C[\uDDEA\uDDEC-\uDDEE\uDDF2\uDDF3\uDDF5\uDDF7\uDDFC\uDDFE\uDDFF])|\uD83C\uDDFE(?:\uD83C[\uDDEA\uDDF9])|\uD83C\uDDEE(?:\uD83C[\uDDE8-\uDDEA\uDDF1-\uDDF4\uDDF6-\uDDF9])|\uD83C\uDDF9(?:\uD83C[\uDDE6\uDDE8\uDDE9\uDDEB-\uDDED\uDDEF-\uDDF4\uDDF7\uDDF9\uDDFB\uDDFC\uDDFF])|\uD83C\uDDEC(?:\uD83C[\uDDE6\uDDE7\uDDE9-\uDDEE\uDDF1-\uDDF3\uDDF5-\uDDFA\uDDFC\uDDFE])|\uD83C\uDDFA(?:\uD83C[\uDDE6\uDDEC\uDDF2\uDDF3\uDDF8\uDDFE\uDDFF])|\uD83C\uDDEA(?:\uD83C[\uDDE6\uDDE8\uDDEA\uDDEC\uDDED\uDDF7-\uDDFA])|\uD83C\uDDFC(?:\uD83C[\uDDEB\uDDF8])|(?:\u26F9|\uD83C[\uDFCB\uDFCC]|\uD83D\uDD75)(?:\uD83C[\uDFFB-\uDFFF])|(?:\uD83C[\uDFC3\uDFC4\uDFCA]|\uD83D[\uDC6E\uDC71\uDC73\uDC77\uDC81\uDC82\uDC86\uDC87\uDE45-\uDE47\uDE4B\uDE4D\uDE4E\uDEA3\uDEB4-\uDEB6]|\uD83E[\uDD26\uDD37-\uDD39\uDD3D\uDD3E\uDDD6-\uDDDD])(?:\uD83C[\uDFFB-\uDFFF])|(?:[\u261D\u270A-\u270D]|\uD83C[\uDF85\uDFC2\uDFC7]|\uD83D[\uDC42\uDC43\uDC46-\uDC50\uDC66\uDC67\uDC70\uDC72\uDC74-\uDC76\uDC78\uDC7C\uDC83\uDC85\uDCAA\uDD74\uDD7A\uDD90\uDD95\uDD96\uDE4C\uDE4F\uDEC0\uDECC]|\uD83E[\uDD18-\uDD1C\uDD1E\uDD1F\uDD30-\uDD36\uDDD1-\uDDD5])(?:\uD83C[\uDFFB-\uDFFF])|\uD83D\uDC68(?:\u200D(?:(?:(?:\uD83D[\uDC68\uDC69])\u200D)?\uD83D\uDC67|(?:(?:\uD83D[\uDC68\uDC69])\u200D)?\uD83D\uDC66)|\uD83C[\uDFFB-\uDFFF])|(?:[\u261D\u26F9\u270A-\u270D]|\uD83C[\uDF85\uDFC2-\uDFC4\uDFC7\uDFCA-\uDFCC]|\uD83D[\uDC42\uDC43\uDC46-\uDC50\uDC66-\uDC69\uDC6E\uDC70-\uDC78\uDC7C\uDC81-\uDC83\uDC85-\uDC87\uDCAA\uDD74\uDD75\uDD7A\uDD90\uDD95\uDD96\uDE45-\uDE47\uDE4B-\uDE4F\uDEA3\uDEB4-\uDEB6\uDEC0\uDECC]|\uD83E[\uDD18-\uDD1C\uDD1E\uDD1F\uDD26\uDD30-\uDD39\uDD3D\uDD3E\uDDD1-\uDDDD])(?:\uD83C[\uDFFB-\uDFFF])?|(?:[\u231A\u231B\u23E9-\u23EC\u23F0\u23F3\u25FD\u25FE\u2614\u2615\u2648-\u2653\u267F\u2693\u26A1\u26AA\u26AB\u26BD\u26BE\u26C4\u26C5\u26CE\u26D4\u26EA\u26F2\u26F3\u26F5\u26FA\u26FD\u2705\u270A\u270B\u2728\u274C\u274E\u2753-\u2755\u2757\u2795-\u2797\u27B0\u27BF\u2B1B\u2B1C\u2B50\u2B55]|\uD83C[\uDC04\uDCCF\uDD8E\uDD91-\uDD9A\uDDE6-\uDDFF\uDE01\uDE1A\uDE2F\uDE32-\uDE36\uDE38-\uDE3A\uDE50\uDE51\uDF00-\uDF20\uDF2D-\uDF35\uDF37-\uDF7C\uDF7E-\uDF93\uDFA0-\uDFCA\uDFCF-\uDFD3\uDFE0-\uDFF0\uDFF4\uDFF8-\uDFFF]|\uD83D[\uDC00-\uDC3E\uDC40\uDC42-\uDCFC\uDCFF-\uDD3D\uDD4B-\uDD4E\uDD50-\uDD67\uDD7A\uDD95\uDD96\uDDA4\uDDFB-\uDE4F\uDE80-\uDEC5\uDECC\uDED0-\uDED2\uDEEB\uDEEC\uDEF4-\uDEF8]|\uD83E[\uDD10-\uDD3A\uDD3C-\uDD3E\uDD40-\uDD45\uDD47-\uDD4C\uDD50-\uDD6B\uDD80-\uDD97\uDDC0\uDDD0-\uDDE6])|(?:[#\*0-9\xA9\xAE\u203C\u2049\u2122\u2139\u2194-\u2199\u21A9\u21AA\u231A\u231B\u2328\u23CF\u23E9-\u23F3\u23F8-\u23FA\u24C2\u25AA\u25AB\u25B6\u25C0\u25FB-\u25FE\u2600-\u2604\u260E\u2611\u2614\u2615\u2618\u261D\u2620\u2622\u2623\u2626\u262A\u262E\u262F\u2638-\u263A\u2640\u2642\u2648-\u2653\u2660\u2663\u2665\u2666\u2668\u267B\u267F\u2692-\u2697\u2699\u269B\u269C\u26A0\u26A1\u26AA\u26AB\u26B0\u26B1\u26BD\u26BE\u26C4\u26C5\u26C8\u26CE\u26CF\u26D1\u26D3\u26D4\u26E9\u26EA\u26F0-\u26F5\u26F7-\u26FA\u26FD\u2702\u2705\u2708-\u270D\u270F\u2712\u2714\u2716\u271D\u2721\u2728\u2733\u2734\u2744\u2747\u274C\u274E\u2753-\u2755\u2757\u2763\u2764\u2795-\u2797\u27A1\u27B0\u27BF\u2934\u2935\u2B05-\u2B07\u2B1B\u2B1C\u2B50\u2B55\u3030\u303D\u3297\u3299]|\uD83C[\uDC04\uDCCF\uDD70\uDD71\uDD7E\uDD7F\uDD8E\uDD91-\uDD9A\uDDE6-\uDDFF\uDE01\uDE02\uDE1A\uDE2F\uDE32-\uDE3A\uDE50\uDE51\uDF00-\uDF21\uDF24-\uDF93\uDF96\uDF97\uDF99-\uDF9B\uDF9E-\uDFF0\uDFF3-\uDFF5\uDFF7-\uDFFF]|\uD83D[\uDC00-\uDCFD\uDCFF-\uDD3D\uDD49-\uDD4E\uDD50-\uDD67\uDD6F\uDD70\uDD73-\uDD7A\uDD87\uDD8A-\uDD8D\uDD90\uDD95\uDD96\uDDA4\uDDA5\uDDA8\uDDB1\uDDB2\uDDBC\uDDC2-\uDDC4\uDDD1-\uDDD3\uDDDC-\uDDDE\uDDE1\uDDE3\uDDE8\uDDEF\uDDF3\uDDFA-\uDE4F\uDE80-\uDEC5\uDECB-\uDED2\uDEE0-\uDEE5\uDEE9\uDEEB\uDEEC\uDEF0\uDEF3-\uDEF8]|\uD83E[\uDD10-\uDD3A\uDD3C-\uDD3E\uDD40-\uDD45\uDD47-\uDD4C\uDD50-\uDD6B\uDD80-\uDD97\uDDC0\uDDD0-\uDDE6])\uFE0F)/;
    function getText(D) {
        var u = D.nodeType,
            F = "";
        if (1 === u || 9 === u || 11 === u) {
            if ("string" == typeof D.textContent) return D.textContent;
            for (D = D.firstChild; D; D = D.nextSibling) F += getText(D);
        } else if (3 === u || 4 === u) return D.nodeValue;
        return F;
    }
    function emojiSafeSplit(D, u, F) {
        if ((F && (D = D.replace(n, "")), u && "" !== u)) return D.replace(/>/g, "&gt;").replace(/</g, "&lt;").split(u);
        for (var C, E, e = [], t = D.length, i = 0; i < t; i++)
            ((55296 <= (E = D.charAt(i)).charCodeAt(0) && E.charCodeAt(0) <= 56319) || (65024 <= D.charCodeAt(i + 1) && D.charCodeAt(i + 1) <= 65039)) &&
                ((C = ((D.substr(i, 12).split(r) || [])[1] || "").length || 2), (E = D.substr(i, C)), (i += C - (e.emoji = 1))),
                e.push(">" === E ? "&gt;" : "<" === E ? "&lt;" : E);
        return e;
    }
    var s =
        ((CharSet.prototype.grow = function grow(D) {
            for (var u = 0; u < 20; u++) this.sets[u] += F(D - this.length, this.chars);
            this.length = D;
        }),
        CharSet);
    function CharSet(D) {
        (this.chars = emojiSafeSplit(D)), (this.sets = []), (this.length = 50);
        for (var u = 0; u < 20; u++) this.sets[u] = F(80, this.chars);
    }
    function i() {
        return u || ("undefined" != typeof window && (u = window.gsap) && u.registerPlugin && u);
    }
    function p() {
        a = u = i();
    }
    var u,
        a,
        B = /\s+/g,
        F = function _scrambleText(D, u) {
            for (var F = u.length, C = ""; -1 < --D; ) C += u[~~(Math.random() * F)];
            return C;
        },
        C = "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        E = C.toLowerCase(),
        l = { upperCase: new s(C), lowerCase: new s(E), upperAndLowerCase: new s(C + E) },
        e = {
            version: "3.1.1",
            name: "scrambleText",
            register: function register(D) {
                (u = D), p();
            },
            init: function init(D, u, F) {
                if ((a || p(), (this.prop = "innerHTML" in D ? "innerHTML" : "textContent" in D ? "textContent" : 0), this.prop)) {
                    (this.target = D), "object" != typeof u && (u = { text: u });
                    var C,
                        E,
                        e,
                        t,
                        i = u.text || u.value,
                        n = !1 !== u.trim,
                        r = this;
                    return (
                        (r.delimiter = C = u.delimiter || ""),
                        (r.original = emojiSafeSplit(getText(D).replace(B, " ").split("&nbsp;").join(""), C, n)),
                        ("{original}" !== i && !0 !== i && null != i) || (i = r.original.join(C)),
                        (r.text = emojiSafeSplit((i || "").replace(B, " "), C, n)),
                        (r.hasClass = !(!u.newClass && !u.oldClass)),
                        (r.newClass = u.newClass),
                        (r.oldClass = u.oldClass),
                        (t = "" === C),
                        (r.textHasEmoji = t && !!r.text.emoji),
                        (r.charsHaveEmoji = !!u.chars && !!emojiSafeSplit(u.chars).emoji),
                        (r.length = t ? r.original.length : r.original.join(C).length),
                        (r.lengthDif = (t ? r.text.length : r.text.join(C).length) - r.length),
                        (r.fillChar = u.fillChar || (u.chars && ~u.chars.indexOf(" ")) ? "&nbsp;" : ""),
                        (r.charSet = e = l[u.chars || "upperCase"] || new s(u.chars)),
                        (r.speed = 0.05 / (u.speed || 1)),
                        (r.prevScrambleTime = 0),
                        (r.setIndex = (20 * Math.random()) | 0),
                        (E = r.length + Math.max(r.lengthDif, 0)) > e.length && e.grow(E),
                        (r.chars = e.sets[r.setIndex]),
                        (r.revealDelay = u.revealDelay || 0),
                        (r.tweenLength = !1 !== u.tweenLength),
                        (r.tween = F),
                        (r.rightToLeft = !!u.rightToLeft),
                        r._props.push("scrambleText", "text"),
                        1
                    );
                }
            },
            render: function render(D, u) {
                var F,
                    C,
                    E,
                    e,
                    t,
                    i,
                    n,
                    r,
                    s,
                    a = u.target,
                    B = u.prop,
                    l = u.text,
                    o = u.delimiter,
                    A = u.tween,
                    h = u.prevScrambleTime,
                    p = u.revealDelay,
                    f = u.setIndex,
                    g = u.chars,
                    c = u.charSet,
                    d = u.length,
                    m = u.textHasEmoji,
                    x = u.charsHaveEmoji,
                    S = u.lengthDif,
                    j = u.tweenLength,
                    w = u.oldClass,
                    v = u.newClass,
                    b = u.rightToLeft,
                    T = u.fillChar,
                    y = u.speed,
                    L = u.original,
                    _ = u.hasClass,
                    M = l.length,
                    H = A._time,
                    I = H - h;
                p && (A._from && (H = A._dur - H), (D = 0 === H ? 0 : H < p ? 1e-6 : H === A._dur ? 1 : A._ease((H - p) / (A._dur - p)))),
                    D < 0 ? (D = 0) : 1 < D && (D = 1),
                    b && (D = 1 - D),
                    (F = ~~(D * M + 0.5)),
                    (e = D ? ((y < I || I < -y) && ((u.setIndex = f = (f + ((19 * Math.random()) | 0)) % 20), (u.chars = c.sets[f]), (u.prevScrambleTime += I)), g) : L.join(o)),
                    (e = b
                        ? 1 !== D || (!A._from && "isFromStart" !== A.data)
                            ? ((n = l.slice(F).join(o)),
                              (E = x
                                  ? emojiSafeSplit(e)
                                        .slice(0, (d + (j ? 1 - D * D * D : 1) * S - (m ? emojiSafeSplit(n) : n).length + 0.5) | 0)
                                        .join("")
                                  : e.substr(0, (d + (j ? 1 - D * D * D : 1) * S - (m ? emojiSafeSplit(n) : n).length + 0.5) | 0)),
                              n)
                            : ((E = ""), L.join(o))
                        : ((E = l.slice(0, F).join(o)),
                          (C = (m ? emojiSafeSplit(E) : E).length),
                          x
                              ? emojiSafeSplit(e)
                                    .slice(C, (d + (j ? 1 - (D = 1 - D) * D * D * D : 1) * S + 0.5) | 0)
                                    .join("")
                              : e.substr(C, (d + (j ? 1 - (D = 1 - D) * D * D * D : 1) * S - C + 0.5) | 0))),
                    (n = _ ? ((t = (r = b ? w : v) && 0 != F) ? "<span class='" + r + "'>" : "") + E + (t ? "</span>" : "") + ((i = (s = b ? v : w) && F !== M) ? "<span class='" + s + "'>" : "") + o + e + (i ? "</span>" : "") : E + o + e),
                    (a[B] = "&nbsp;" === T && ~n.indexOf("  ") ? n.split("  ").join("&nbsp;&nbsp;") : n);
            },
        };
    (e.emojiSafeSplit = emojiSafeSplit), (e.getText = getText), i() && u.registerPlugin(e), (D.ScrambleTextPlugin = e), (D.default = e);
    if (typeof window === "undefined" || window !== D) {
        Object.defineProperty(D, "__esModule", { value: !0 });
    } else {
        delete D.default;
    }
});
