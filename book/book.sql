drop table if exists tb_abt_imdb;
create table tb_abt_imdb as

with tb_titulos as (
    select title_id,
        count(title) as qt_titulos
    from akas
    GROUP by title_id
)


SELECT t1.title_id,
       case when t1.type = 'tvMovie' then 1 else 0 end as filme_tv,
       t1.is_adult,
       t1.premiered as ano_estreia,
       t1.runtime_minutes as tempo_duracao,
       case when t1.genres like ('%action%') then 1 else 0 end as genero_action,
       case when t1.genres like ('%adult%') then 1 else 0 end as genero_adult,
       case when t1.genres like ('%adventure%') then 1 else 0 end as genero_adventure,
       case when t1.genres like ('%animation%') then 1 else 0 end as genero_animation,
       case when t1.genres like ('%biography%') then 1 else 0 end as genero_biography,
       case when t1.genres like ('%comedy%') then 1 else 0 end as genero_comedy,
       case when t1.genres like ('%crime%') then 1 else 0 end as genero_crime,
       case when t1.genres like ('%documentary%') then 1 else 0 end as genero_documentary,
       case when t1.genres like ('%drama%') then 1 else 0 end as genero_drama,
       case when t1.genres like ('%family%') then 1 else 0 end as genero_family,
       case when t1.genres like ('%fantasy%') then 1 else 0 end as genero_fantasy,
       case when t1.genres like ('%film-noir%') then 1 else 0 end as genero_filmnoir,
       case when t1.genres like ('%game-show%') then 1 else 0 end as genero_gameshow,
       case when t1.genres like ('%history%') then 1 else 0 end as genero_history,
       case when t1.genres like ('%horror%') then 1 else 0 end as genero_horror,
       case when t1.genres like ('%music%') then 1 else 0 end as genero_music,
       case when t1.genres like ('%musical%') then 1 else 0 end as genero_musical,
       case when t1.genres like ('%mystery%') then 1 else 0 end as genero_mystery,
       case when t1.genres like ('%news%') then 1 else 0 end as genero_news,
       case when t1.genres like ('\N') then 1 else 0 end as genero_nulo,
       case when t1.genres like ('%reality-tv%') then 1 else 0 end as genero_realitytv,
       case when t1.genres like ('%romance%') then 1 else 0 end as genero_romance,
       case when t1.genres like ('%sci-fi%') then 1 else 0 end as genero_scifi,
       case when t1.genres like ('%short%') then 1 else 0 end as genero_short,
       case when t1.genres like ('%talk-show%') then 1 else 0 end as genero_talkshow,
       case when t1.genres like ('%thriller%') then 1 else 0 end as genero_thriller,
       case when t1.genres like ('%war%') then 1 else 0 end as genero_war,
       case when t1.genres like ('%western%') then 1 else 0 end as genero_western,
       t2.qt_titulos as numero_titulos,
       count(DISTINCT t3.person_id) as qt_crew,
       t4.votes as num_votos,
       t4.rating
       
       
FROM titles as t1

left join tb_titulos as t2
on t1.title_id = t2.title_id

left join crew as t3
on t1.title_id = t3.title_id

left join ratings as t4
on t1.title_id = t4.title_id

where t1.type in ('tvMovie','movie')
and rating IS NOT NULL
and tempo_duracao <= 360
and num_votos > 20

group by t1.title_id;
