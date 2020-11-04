<template>
  <span @click="deleteEditor({editorType, index})">
    <i class="delete-button fa fa-trash"/>
  </span>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import { mapMutations } from 'vuex';
import EditorType from '../../EditorType';

@Component({
  methods: mapMutations(['deleteEditor']),
})
export default class DeleteEditorButton extends Vue {
  @Prop({ required: true }) readonly editorType!: EditorType;
  @Prop({ required: true }) readonly index!: number;
}
</script>

<style lang="scss" scoped>
  .delete-button {
    color: var(--foreground);
    background: none;
    padding: 0 4px 2px;
    border-radius: 2px;
    &:hover {
      background: var(--blue);
    }
  }
</style>
